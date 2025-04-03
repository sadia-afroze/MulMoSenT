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

from transformers import ViTModel, ElectraModel, ElectraTokenizer, get_linear_schedule_with_warmup,CLIPModel, CLIPProcessor, AutoConfig, AdamW

label_map = {"Business": 0, "Crime": 1, "Entertainment": 2, "Environment": 3, "Science-Tech": 4, "Others": 5}

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

class CustomDataLoaderMMTC(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.image_path = self.data["Image_path"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = 224
        self.tokenizer = ElectraTokenizer.from_pretrained("csebuetnlp/banglabert")
        # self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # self.tokenizer = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",load_in_8bit=True)
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        image = self.transform(image)
        text = self.sentences[idx]
        label = self.label[idx]
        label = label_map[label]
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=14, return_tensors='pt')
        return {'image': image, 'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'image_path':self.image_path[idx], 'label': torch.tensor(label,dtype=torch.long)}

class CustomImageClassification(torch.nn.Module):
    def __init__(self, num_labels=6):
        super(CustomImageClassification, self).__init__()
        self.config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32", num_labels=6)
        self.config.return_dict = True
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", return_dict=True)
        self.text_embedding_size = 512
        self.image_embedding_size = 512
        self.input_size = 50271
        self.fc = torch.nn.Linear(self.input_size + 1, self.text_embedding_size)
        self.linear = torch.nn.Linear(self.text_embedding_size, self.config.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pixel_values, attention_mask, input_ids):
        output = self.model(pixel_values=pixel_values, attention_mask=attention_mask, input_ids=input_ids)
        # For CLIP and BLIP
        image_embeds = output["image_embeds"]
        image_embeds = image_embeds.to(dtype=torch.float32)
        image_embeds = self.dropout(image_embeds)
        return image_embeds


class CustomTextClassification(torch.nn.Module):
    def __init__(self, num_labels=6):
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
        output = self.dropout(output.last_hidden_state[:, 0, :])
        return output
        #output = self.linear(output)
        #output = self.softmax(output)
        #return output



class CustoMultiModalImgaeTextClassification(torch.nn.Module):
    def __init__(self, num_labels=6, image_model=None, text_model=None):
        super(CustoMultiModalImgaeTextClassification, self).__init__()
        self.hidden_size = 768
        self.num_labels = num_labels
        self.image_model = image_model
        self.text_model = text_model
        self.linear = torch.nn.Linear(self.hidden_size+512, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values, input_ids, attention_mask):
        image_embeds = self.image_model(pixel_values, attention_mask, input_ids)
        text_embeds = self.text_model(input_ids, attention_mask)
        output = torch.cat((image_embeds, text_embeds), dim=1)
        output = self.linear(output)
        output = self.softmax(output)
        return output


def TestMMTC(temp_model):
    #best_model_path = './mmbtc-6-model/best_mmtc_model_withoutfreez.pt'
    dataset = CustomDataLoaderMMTC("./BMMTC6-Final/BMMTC6-Test/test.csv")
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = temp_model #torch.load(best_model_path).to(device)
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
    datasets = CustomDataLoaderMMTC("./BMMTC6-Final/BMMTC6-Train/train.csv")
    dataloader = DataLoader(datasets, batch_size=32, shuffle=True)
    num_classes = 6
    image_model = CustomImageClassification(num_labels=num_classes)
    image_model = image_model.to(device)
    text_model = CustomTextClassification(num_labels=num_classes)
    text_model = text_model.to(device)
    model = CustoMultiModalImgaeTextClassification(image_model=image_model, text_model=text_model)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-6)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    epochs = 30
    class_counts = [2335, 1000, 3667, 762, 1134, 1784]
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
            output = model(images, input_ids, attention_mask)
            labels = torch.nn.functional.one_hot(labels, num_classes=6).to(device)
            loss = criterion(output, labels.float())
            epoch_loss += loss.item()/batch["image"].shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.argmax(dim=1).cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(labels)
        #scheduler.step()
        print("Epoch: ", epoch, "Loss: ", epoch_loss)
        acc = TestMMTC(model)
        #print(classification_report(actual_labels, predict_labels))
        if acc > globAlaccuracy:
            globAlaccuracy = acc
            best_model = model
            torch.save(best_model, "./mmbtc-6-model/BMMTC_Focal_loss_CLIP_Image_BanglaBERT.pt")


if __name__ == "__main__":
    TrainMMTC()
    #TestMMTC()