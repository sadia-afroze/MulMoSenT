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

from transformers import AutoTokenizer, MT5Model, ViTModel, get_linear_schedule_with_warmup

label_map = {"Business": 0, "Crime": 1, "Entertainment": 2, "Environment": 3, "Science-Tech": 4, "Others": 5}



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





class CustomTextDataLoader(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.image_path = self.data["Image_path"].tolist()
        self.samples = len(self.data)
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = 224

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        text = self.sentences[idx]
        image = cv2.imread(self.image_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = self.transform(image)
        label = self.label[idx]
        label = label_map[label]
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors='pt')
        return {'input_ids': encoded_input['input_ids'].flatten(), 'image': image, 'attention_mask': encoded_input['attention_mask'].flatten(), 'label': torch.tensor(label,dtype=torch.long)}



class mT5_ViT_MFB(nn.Module):
    def __init__(self, num_labels=6):
        super(mT5_ViT_MFB, self).__init__()
        self.text_embeddings_size = 768
        self.image_embeddings_size = 768
        self.num_labels = num_labels
        self.model_text = MT5Model.from_pretrained("google/mt5-base", return_dict=True)
        self.model_image = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", return_dict=True)
        self.image_embed_dim = 768
        self.text_embed_dim = 768
        self.fact_dim = 16
        self.output_dim = 1000
        self.num_labels = num_labels
        self.mfb_fusion = MFBFusion(image_embed_dim=self.image_embed_dim, text_embed_dim=self.text_embed_dim,
                                    fact_dim=self.fact_dim, output_dim=self.output_dim)
        self.linear = torch.nn.Linear(self.output_dim, self.num_labels)

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, images, input_ids, attention_mask, decoder_input_ids=None, decoder_inputs_embeds=None):
        encoder_outputs = self.model_text.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_output = encoder_outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        image_output = self.model_image(pixel_values=images)
        image_output = image_output.last_hidden_state[:, 0, :]
        mfb_output = self.mfb_fusion(text_output, image_output)
        mfb_output = self.linear(mfb_output)
        mfb_output = self.softmax(mfb_output)
        return mfb_output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def TestTextModel(temp_model):
    datasets = CustomTextDataLoader("./BMMTC6-Final/test.csv")
    dataloader = DataLoader(datasets, batch_size=16, shuffle=False)
    temp_model.eval()
    pred_labels = []
    actual_labels = []
    for i, data in enumerate(dataloader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['label'].to(device)
        images = data['image'].to(device)
        output = temp_model(images, input_ids, attention_mask)
        _, pred = torch.max(output, 1)
        pred_labels.extend(pred.cpu().numpy())
        actual_labels.extend(label.cpu().numpy())
    acc = accuracy_score(actual_labels, pred_labels)
    print("Accuracy: ", accuracy_score(actual_labels, pred_labels))
    print("Precision: ", precision_score(actual_labels, pred_labels, average='macro'))
    print("Recall: ", recall_score(actual_labels, pred_labels, average='macro'))
    print("F1 Score: ", f1_score(actual_labels, pred_labels, average='macro'))
    print("Classification Report: \n", classification_report(actual_labels, pred_labels))
    print("Confusion Matrix: \n", confusion_matrix(actual_labels, pred_labels))
    return acc

def TrainTextModel():
    datasets = CustomTextDataLoader("./BMMTC6-Final/train.csv")
    dataloader = DataLoader(datasets, batch_size=16, shuffle=True)
    model = mT5_ViT_MFB()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epoch = 20
    total_steps = len(dataloader) * 10
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()
    globalACC = 0
    for epoch in range(epoch):
        total_loss = 0
        for i, data in enumerate(dataloader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            images = data['image'].to(device)
            label = data['label'].to(device)
            optimizer.zero_grad()
            output = model(images, input_ids, attention_mask)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print("Epoch: ", epoch, "Loss: ", total_loss)
        acc = TestTextModel(model)
        if acc > globalACC:
            globalACC = acc
            torch.save(model.state_dict(), './Model/mT5_ViT.pth')


if __name__ == "__main__":
    TrainTextModel()