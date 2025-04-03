import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import re
import math

from torchgen.native_function_generation import self_to_out_signature
from torchvision import transforms, utils
import torch.nn.functional as F
import torch.nn as nn
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, euclidean_distances

from transformers import ViTModel, ElectraModel, ElectraTokenizer, get_linear_schedule_with_warmup, AutoTokenizer,AutoModel
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import attention

label_map = {"Business": 0, "Crime": 1, "Entertainment": 2, "Environment": 3, "Science-Tech": 4, "Others": 5}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ArcLoss(nn.Module):
    def __init__(self, s=30.0, m=0.50):
        super(ArcLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, label):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss = self.criterion(output, label)
        return loss


#Label,Text,Image_path
class CustomDataLoaderMMTC(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.image_path = self.data["Image_path"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = 224
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = self.transform(image)
        text = self.sentences[idx]
        label = self.label[idx]
        label = label_map[label]
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors='pt')
        return {'image': image, 'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'image_path':self.image_path[idx], 'label': torch.tensor(label,dtype=torch.long)}

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



class CustomImageClassification(torch.nn.Module):
    def __init__(self, num_labels=6):
        super(CustomImageClassification, self).__init__()
        # self.config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=6)
        # # Return CLS token
        # self.config.return_dict = True
        self.hidden_size = 768
        self.num_labels = 6
        self.model = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k", return_dict=True)
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.ReLu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values):
        output = self.model(pixel_values=pixel_values)
        output = output.last_hidden_state[:, 0, :]
        return output



class CustomTextClassification(torch.nn.Module):
    def __init__(self, num_labels=6):
        super(CustomTextClassification, self).__init__()
        self.hidden_size = 768
        self.num_labels = num_labels
        self.model = AutoModel.from_pretrained("xlm-roberta-base", return_dict=True)
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.ReLu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state[:, 0, :]
        return output



class CustoMultiModalImgaeTextClassification(torch.nn.Module):
    def __init__(self, num_labels=6, image_model=None, text_model=None):
        super(CustoMultiModalImgaeTextClassification, self).__init__()
        self.hidden_size = 768
        self.num_labels = num_labels
        self.image_model = image_model
        self.text_model = text_model
        self.num_heads = 1
        self.image_embed_dim = 768
        self.text_embed_dim = 768
        self.fact_dim = 16
        self.output_dim = 1000
        self.mfb_fusion = MFBFusion(image_embed_dim=self.image_embed_dim, text_embed_dim=self.text_embed_dim, fact_dim=self.fact_dim, output_dim=self.output_dim)
        self.linear = torch.nn.Linear(self.output_dim+self.image_embed_dim+self.text_embed_dim, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values, input_ids, attention_mask):
        image_output = self.image_model(pixel_values)
        text_output = self.text_model(input_ids, attention_mask)
        text_image_fusion = torch.cat((image_output, text_output), dim=1)
        mfb_fusion_output = self.mfb_fusion(text_output, image_output)
        mfb_fusion_output = torch.cat((mfb_fusion_output, text_image_fusion), dim=1)
        output = self.linear( mfb_fusion_output )
        output = self.softmax(output)
        return output


def TestMMTC():
    dataset = CustomDataLoaderMMTC("./BMMTC6-Final/test.csv")
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torch.load("./Model/XML_ViT.pt")
    model.eval()
    predict_labels = []
    actual_labels = []
    Image_path_list = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            image_path = batch['image_path']
            output = model(images, input_ids, attention_mask)
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            predict_labels.extend(output)
            actual_labels.extend(labels)

            Image_path_list.extend(image_path)
    acc = accuracy_score(actual_labels, predict_labels)
    print(f"Accuracy: {acc}")
    print(classification_report(actual_labels, predict_labels))
    print(confusion_matrix(actual_labels, predict_labels))
    return acc

def TrainMMTC():
    datasets = CustomDataLoaderMMTC("./BMMTC6-Final/train.csv")
    dataloader = DataLoader(datasets, batch_size=32, shuffle=True)
    num_classes = 6
    image_model = CustomImageClassification(num_labels=num_classes)
    image_model = image_model.to(device)
    text_model = CustomTextClassification(num_labels=num_classes)
    text_model = text_model.to(device)
    model = CustoMultiModalImgaeTextClassification(image_model=image_model, text_model=text_model)
    model = model.to(device)
    #model = torch.load("./Model/BMMTC_Focal_loss_Image_Text_MFB_lr2e-5_1000dimReLU.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 10)
    epochs = 20
    class_counts = [2335, 1000, 3667, 762, 1134, 1784]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
    model.train()
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
            output= model(images, input_ids, attention_mask)
            labels = torch.nn.functional.one_hot(labels, num_classes=6).to(device)
            loss = criterion(output, labels.float())
            loss = loss
            epoch_loss += loss.item()/batch["image"].shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.argmax(dim=1).cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(labels)
        scheduler.step()
        print(f"Epoach: {epoch}, Loss: {epoch_loss}, Accuracy: {accuracy_score(actual_labels, predict_labels)}")
        acc = TestMMTC(model)
        if acc > globAlaccuracy:
            globAlaccuracy = acc
            best_model = model
            torch.save(best_model, "./Model/XML_ViT.pt")



if __name__ == "__main__":
    #TrainMMTC()
    TestMMTC()