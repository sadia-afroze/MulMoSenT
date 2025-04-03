import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import re
from torchvision import transforms, utils
import cv2
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, AdamW, get_linear_schedule_with_warmup, AutoConfig, ElectraModel, ElectraTokenizer

label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}

class CustomDataLoaderMMTC(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.image_path = self.data["Image"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.image_size = 224
        self.tokenizer = ElectraTokenizer.from_pretrained("csebuetnlp/banglabert")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("Converted to RGB")
        image = self.transform(image)
        text = self.sentences[idx]
        label = self.label[idx]
        label = label_map[label]
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors='pt')
        return {'image': image, 'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'image_path':self.image_path[idx], 'label': torch.tensor(label,dtype=torch.long)}


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

        return torch.cat((text_to_image_attn,image_to_text_attn), dim=1)

        # Combine the attended features
        #print(f"text_to_image_attn shape: {text_to_image_attn.shape}")
        #print(f"image_to_text_attn shape: {image_to_text_attn.shape}")
        #combined_features = torch.cat((text_to_image_attn, image_to_text_attn), dim=-1)

        #return combined_features


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embeddings):
        if embeddings.dim() == 2:  # If input is 2D, add a batch dimension
            embeddings = embeddings.unsqueeze(0)  # Shape (1, seq_len, embed_dim)

        query = self.query(embeddings)  # Shape (batch_size, seq_len, embed_dim)
        key = self.key(embeddings)  # Shape (batch_size, seq_len, embed_dim)
        value = self.value(embeddings)  # Shape (batch_size, seq_len, embed_dim)

        # Compute attention scores
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # Shape (batch_size, seq_len, seq_len)
        attention_weights = self.softmax(attention_scores / (self.embed_dim ** 0.5))  # Scale by sqrt(embed_dim)

        # Compute output
        output = torch.bmm(attention_weights, value)  # Shape (batch_size, seq_len, embed_dim)

        return output

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


class MFBFusion_Block_sum_pool(nn.Module):
    def __init__(self, image_embed_dim, text_embed_dim, fact_dim, output_dim):
        super(MFBFusion_Block_sum_pool, self).__init__()
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = text_embed_dim
        self.fact_dim = fact_dim
        self.output_dim = output_dim
        self.kernel_size = 2
        self.stride = 1
        self.image_linear = nn.Linear(self.image_embed_dim, self.fact_dim * self.output_dim)
        self.text_linear = nn.Linear(self.text_embed_dim, self.fact_dim * self.output_dim)

    def forward(self, text_embeddings, image_embeddings):
        image_proj = self.image_linear(image_embeddings)
        text_proj = self.text_linear(text_embeddings)
        mfb_output = image_proj * text_proj

        # Reshape to (batch_size, fact_dim, output_dim)
        mfb_output = mfb_output.view(-1, self.fact_dim, self.output_dim)

        # Sum pooling with kernel size and stride
        mfb_output = F.avg_pool1d(mfb_output, kernel_size=self.kernel_size, stride=self.stride) * self.kernel_size

        mfb_output = mfb_output.mean(dim=1)
       # print(f"mfb_output shape: {mfb_output.shape}")

        # Normalize (optional)
        mfb_output = F.normalize(mfb_output, p=2, dim=1)

        return mfb_output


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


class CustomImageClassification(torch.nn.Module):
    def __init__(self, num_labels=3):
        super(CustomImageClassification, self).__init__()
        self.config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32", num_labels=3)
        self.config.return_dict = True
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", return_dict=True)
        self.text_embedding_size = 768
        self.image_embedding_size = 768
        self.input_size =  50271
        self.fc = torch.nn.Linear(self.input_size+1, self.text_embedding_size)
        self.linear = torch.nn.Linear(self.text_embedding_size, self.config.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values, attention_mask, input_ids):
        output = self.model(pixel_values=pixel_values, attention_mask=attention_mask, input_ids=input_ids)
        #For CLIP and BLIP
        image_embeds = output["image_embeds"]
        text_embeds = output["text_embeds"]
        image_embeds = image_embeds.to(dtype=torch.float32)
        text_embeds = text_embeds.to(dtype=torch.float32)
        fused = image_embeds #torch.cat((image_embeds, text_embeds), dim=1)
        fused = self.dropout(fused)
        return fused

class CustomTextClassification(torch.nn.Module):
    def __init__(self, num_labels=3):
        super(CustomTextClassification, self).__init__()
        self.hidden_size = 768
        self.num_labels = num_labels
        self.model = ElectraModel.from_pretrained("csebuetnlp/banglabert", return_dict=True)
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
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
        #self.mfb_block_avg_fusion = MFBFusion_Block_sum_pool(image_embed_dim=self.image_embed_dim, text_embed_dim=self.text_embed_dim, fact_dim=self.fact_dim, output_dim=self.output_dim)
        self.cross_attention = CrossAttention(self.hidden_size, self.num_heads)
        self.mfb_self_attention = SelfAttention(self.output_dim)
        self.image_text_attention = SelfAttention(self.text_embed_dim*2)
        self.linear = torch.nn.Linear(self.output_dim+self.image_embed_dim+self.text_embed_dim, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values, input_ids, attention_mask):
        image_output = self.image_model(pixel_values, attention_mask, input_ids)
        text_output = self.text_model(input_ids, attention_mask)
        text_image_fusion = torch.cat((image_output, text_output), dim=1)
        mfb_fusion_output = self.mfb_fusion(text_output, image_output)
        #mfb_fusion_output = self.mfb_fusion(text_output, image_output)
        mfb_fusion_output = torch.cat((mfb_fusion_output, text_image_fusion), dim=1)
        #mfb_quality = self.embeeds_quality.get_quality_score(mfb_fusion_output)
        output = self.linear( mfb_fusion_output )
        output = self.softmax(output)
        return output




def TestMMTC():
    #best_model_path = './mmbtc-6-model/best_mmtc_model_withoutfreez.pt'
    dataset = CustomDataLoaderMMTC("./test.csv")
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torch.load('./mmbtc-6-model/BMMTC_Focal_loss_Image_Text_MFB_lr2e-5_1000dim.pt').to(device)
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
            output, quality = model(images, input_ids, attention_mask)
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
    datasets = CustomDataLoaderMMTC("./train.csv")
    dataloader = DataLoader(datasets, batch_size=32, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    image_model = CustomImageClassification(num_labels=num_classes)
    image_model = image_model.to(device)
    text_model = CustomTextClassification(num_labels=num_classes)
    text_model = text_model.to(device)
    model = CustoMultiModalImgaeTextClassification(image_model=image_model, text_model=text_model)
    #model = torch.load("./mmbtc-6-model/BMMTC_Focal_loss_Text_guided.pt")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 10)
    epochs = 30
    class_counts = [2335, 1000, 3667]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
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
            output, quality = model(images, input_ids, attention_mask)
            labels = torch.nn.functional.one_hot(labels, num_classes=3).to(device)
            loss = criterion(output, labels.float())
            loss = loss * quality
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
            torch.save(best_model, "./mmbtc-6-model/BMMTC_Focal_loss_Image_Text_MFB_lr2e-5_1000dim.pt")



if __name__ == "__main__":
    TrainMMTC()
    #TestMMTC()
