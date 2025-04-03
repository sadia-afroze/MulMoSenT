import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import re
import torch.nn.functional as F
#from torch.xpu import device
from torchvision import transforms, utils
import cv2
from torch.cuda.amp import GradScaler, autocast
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import BridgeTowerModel, BridgeTowerProcessor, AdamW, get_linear_schedule_with_warmup, AutoConfig

label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}


class ImageTextDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["ReLabel"].tolist()
        self.image_path = self.data["Image"].tolist()
        self.text = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = 224
        self.tokenizer = BridgeTowerProcessor.from_pretrained('BridgeTower/bridgetower-base')

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = self.transform(image)
        text = self.text[idx]
        label = self.label[idx]
        label = label_map[label]
        encoded_input = self.tokenizer(images=image, text=text, padding='max_length', truncation=True, max_length=16,
                                   return_tensors='pt')
        return {
        'image': encoded_input['pixel_values'].squeeze(0),
        'input_ids': encoded_input['input_ids'].squeeze(0),
        'attention_mask': encoded_input['attention_mask'].squeeze(0),
        'label': torch.tensor(label, dtype=torch.long)
        }


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




class BridgeTower(torch.nn.Module):
    def __init__(self, num_labels=3):
        super(BridgeTower, self).__init__()
        self.config = AutoConfig.from_pretrained("BridgeTower/bridgetower-base", num_labels=3)
        self.config.return_dict = True
        self.model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base", return_dict=True)
        self.text_embedding_size = 768
        self.image_embedding_size = 768
        self.input_size =  50271

        self.image_embed_dim = 768
        self.text_embed_dim = 768
        self.fact_dim = 16
        self.output_dim = 1000
        self.num_labels = num_labels
        self.mfb_fusion = MFBFusion(image_embed_dim=self.image_embed_dim, text_embed_dim=self.text_embed_dim,
                                    fact_dim=self.fact_dim, output_dim=self.output_dim)
        self.linear = torch.nn.Linear(self.image_embed_dim+self.text_embed_dim, self.num_labels)



        self.fc = torch.nn.Linear(self.input_size+1, self.text_embedding_size)
        #self.linear = torch.nn.Linear(self.text_embedding_size+self.image_embedding_size, self.config.num_labels)
        #self.linear_image_or_text_only = torch.nn.Linear(self.text_embedding_size, self.config.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, image, attention_mask, input_ids):
        output = self.model(pixel_values=image, input_ids=input_ids, attention_mask=attention_mask)
        image_embeds = output.image_features[:,0,:]
        text_embeds = output.text_features[:,0,:]
        # image_embeds = self.activation(image_embeds)
        # text_embeds = self.activation(text_embeds)
        catFeature = torch.cat((image_embeds, text_embeds), dim=1)
        #fused = self.mfb_fusion(text_embeds, image_embeds)
        output = self.linear(catFeature)
        output = self.softmax(output)
        return output




def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BridgeTower()
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    dataset = ImageTextDataset('./train.csv')
    dataloader = DataLoader(dataset, batch_size=14, shuffle=True)
    class_counts = [5713, 4027, 6003]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)
    certian_loss = torch.nn.CrossEntropyLoss() #FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
    epochs = 15
    globalLoss = 10000000000
    for epoch in range(epochs):
        epoch_loss = 0
        actual_labels = []
        predict_labels = []
        for data in dataloader:
            image = data['image'].to(device)
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['label']
            optimizer.zero_grad()
            outputs = model(image, attention_mask, input_ids)
            label = torch.nn.functional.one_hot(label, num_classes=3).to(device)
            loss =certian_loss(outputs, label.float())
            loss.backward()
            optimizer.step()
            output = outputs.argmax(dim=1)
            label = label.argmax(dim=1)
            output = output.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(label)
            epoch_loss += loss.item()

        acc = accuracy_score(actual_labels, predict_labels)
        print(f"Epoch: {epoch} Loss: {epoch_loss} Accuracy: {acc}")
        scheduler.step()
        if epoch_loss < globalLoss:
            globalLoss = epoch_loss
            torch.save(model.state_dict(), './Model/BridgeTower.pth')
            print('Model Saved')


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BridgeTower()
    model.load_state_dict(torch.load('./Model/BridgeTower.pth'))
    model.to(device)
    model.eval()
    dataset = ImageTextDataset('./test.csv')
    dataloader = DataLoader(dataset, batch_size=14, shuffle=False)
    actual_labels = []
    predict_labels = []
    with torch.no_grad():
        for data in dataloader:
            image = data['image'].to(device)
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['label']
            output = model(image, attention_mask, input_ids)
            output = output.argmax(dim=1)
            output = output.detach().cpu().numpy()
            label = label.numpy()
            predict_labels.extend(output)
            actual_labels.extend(label)
    acc = accuracy_score(actual_labels, predict_labels)
    print(f"Accuracy: {acc}")
    print(f"{classification_report(actual_labels, predict_labels)}")
    print(f"Confusion Matrix:\n {confusion_matrix(actual_labels, predict_labels)}")


if __name__ == '__main__':
    train()
    #test()