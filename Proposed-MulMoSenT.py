import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import re
from torchvision import transforms, utils
import cv2
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import ViTModel, ElectraModel, ElectraTokenizer, get_linear_schedule_with_warmup

label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            targets = targets.long()  # Ensure targets are long tensor for indexing
            alpha = self.alpha[targets]  # Index alpha using targets
            alpha = alpha.view(-1, 1)  # Adjust dimensions for broadcasting
            #print(f"alpha: {alpha.shape}, focal_loss:{focal_loss.shape}")

            focal_loss = alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



class MFBFusion(nn.Module):
    def __init__(self, image_embed_dim, text_embed_dim, fact_dim, output_dim):
        super(MFBFusion, self).__init__()
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = text_embed_dim
        self.fact_dim = fact_dim
        self.output_dim = output_dim
        self.image_linear = nn.Linear(self.image_embed_dim, self.fact_dim * self.output_dim)
        self.text_linear = nn.Linear(self.text_embed_dim, self.fact_dim * self.output_dim)
    def forward(self, text_embeddings, image_embeddings):
        image_proj = self.image_linear(image_embeddings)
        text_proj = self.text_linear(text_embeddings)
        mfb_output = image_proj * text_proj
        mfb_output = mfb_output.view(-1,self.fact_dim, self.output_dim)
        mfb_output = mfb_output.mean(dim=1)
        mfb_output = F.normalize(mfb_output, p=2, dim=1)
        return mfb_output

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = 768
        self.num_heads = 1
        self.text_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        self.image_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)

    def forward(self, text_embeddings, image_embeddings):
        # text_embeddings and image_embeddings shape: (batch_size, seq_length, embed_dim)

        # Apply text-to-image attention
        text_to_image_attn, _ = self.text_attention(text_embeddings, image_embeddings, image_embeddings)

        # Apply image-to-text attention
        image_to_text_attn, _ = self.image_attention(image_embeddings, text_embeddings, text_embeddings)

        return text_to_image_attn

        # Combine the attended features
        #print(f"text_to_image_attn shape: {text_to_image_attn.shape}")
        #print(f"image_to_text_attn shape: {image_to_text_attn.shape}")
        #combined_features = torch.cat((text_to_image_attn, image_to_text_attn), dim=-1)

        #return combined_features



class CustomImageClassification(torch.nn.Module):
    def __init__(self, num_labels=3):
        super(CustomImageClassification, self).__init__()
        self.hidden_size = 768
        self.num_labels = 3
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", return_dict=True)
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values):
        output = self.model(pixel_values=pixel_values)
        output = self.dropout(output.last_hidden_state[:, 0, :])
        #output = self.linear(self.activation(output))
        #output = self.softmax(output)
        #return output
        #print(output.shape)
        #output = self.linear(output)
        #output = self.softmax(output)
        #print(output)
        return output



class CustomTextClassification(torch.nn.Module):
    def __init__(self, num_labels=3):
        super(CustomTextClassification, self).__init__()
        self.hidden_size = 768
        self.num_labels = num_labels
        self.model = ElectraModel.from_pretrained("csebuetnlp/banglabert", return_dict=True)
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(output.last_hidden_state[:, 0, :])
        return output
        #output = self.linear(output)
        #output = self.softmax(output)
        #return output


class CustomDataLoaderMMTC(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["ReLabel"].tolist()
        self.image_path = self.data["Image"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = 224
        self.tokenizer = ElectraTokenizer.from_pretrained("csebuetnlp/banglabert")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("Converted to RGB")
        image = image.resize((self.image_size, self.image_size))
        image = self.transform(image)
        text = self.sentences[idx]
        label = self.label[idx]
        label = label_map[label]
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        return {'image': image, 'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'image_path':self.image_path[idx], 'label': torch.tensor(label,dtype=torch.long),'text':text}

class CustoMultiModalImgaeTextClassification(torch.nn.Module):
    def __init__(self, num_labels=3, image_model=None, text_model=None):
        super(CustoMultiModalImgaeTextClassification, self).__init__()
        self.hidden_size = 768
        self.num_labels = num_labels
        self.image_model = image_model
        self.text_model = text_model

        self.num_heads = 1
        self.embed_dim = 768

        self.image_embed_dim = 768
        self.text_embed_dim = 768
        self.fact_dim = 16
        self.output_dim = 1000
        self.mfb_fusion = MFBFusion(image_embed_dim=self.image_embed_dim, text_embed_dim=self.text_embed_dim,
                                    fact_dim=self.fact_dim, output_dim=self.output_dim)

        self.cross_attention = CrossAttention(self.embed_dim, self.num_heads)

        self.linear = torch.nn.Linear(self.embed_dim + self.image_embed_dim, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values, input_ids, attention_mask):
        image_output = self.image_model(pixel_values)
        text_output = self.text_model(input_ids, attention_mask)
        cross_text_to_image_attn = self.cross_attention(text_output, image_output)
        #mfb_output = self.mfb_fusion(text_output, image_output)
        #Average the image and text embeddings
        avg_fusion = (image_output + text_output) / 2.0
        output = torch.cat((avg_fusion, cross_text_to_image_attn), dim=1)
        output = self.linear(output)
        output = self.softmax(output)
        return output


def TestMMTC(temp_model):
    best_model_path = './Ablation_Model/MSA_BanglaBERT_ViT_avg_cross_attention_Focall_dr-0.pt'
    dataset = CustomDataLoaderMMTC("./test.csv")
    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model =  temp_model #torch.load(best_model_path).to(device)
    model.eval()
    predict_labels = []
    actual_labels = []
    Image_path_list = []
    text_list = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            image_path = batch['image_path']
            text_array = batch['text']
            output = model(images, input_ids, attention_mask)
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(labels)
            Image_path_list.extend(image_path)
            text_list.extend(text_array)

    acc = accuracy_score(actual_labels, predict_labels)
    print("Accuracy: ", acc)
    print(classification_report(actual_labels, predict_labels))
    print(confusion_matrix(actual_labels, predict_labels))
    return acc




def TrainMMTC():
    datasets = CustomDataLoaderMMTC("./train.csv")
    dataloader = DataLoader(datasets, batch_size=32, shuffle=True)
    num_classes = 3
    image_model = CustomImageClassification(num_labels=num_classes)
    image_model = image_model.to(device)
    text_model = CustomTextClassification(num_labels=num_classes)
    text_model = text_model.to(device)

    model = CustoMultiModalImgaeTextClassification(num_labels=num_classes, image_model=image_model, text_model=text_model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 10)
    epochs = 20
    class_counts = [5713, 4027, 6003]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
    certiation_loss = criterion #torch.nn.CrossEntropyLoss()
    model.train()
    best_loss = 10000000000
    best_model = None
    globAlaccuracy = 0
    for epoch in range(epochs):
        epoch_loss = 0
        predict_labels = []
        actual_labels = []
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            #image_path = batch['image_path']
            #print(image_path)
            output = model(images, input_ids, attention_mask)
            labels = torch.nn.functional.one_hot(labels, num_classes=3).to(device)
            loss = certiation_loss(output, labels.float())
            epoch_loss += loss.item()/batch["image"].shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.argmax(dim=1).cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(labels)
        scheduler.step()
        print("Epoch: ", epoch, "Loss: ", epoch_loss)
        acc = TestMMTC(model)
        #print(classification_report(actual_labels, predict_labels))
        if acc > globAlaccuracy:
            globAlaccuracy = acc
            best_model = model
            torch.save(best_model, "./Ablation_Model/MSA_BanglaBERT_ViT_avg_cross_attention_Focall_dr-3.pt")



if __name__ == "__main__":
    #TestMMTC()
    TrainMMTC()
