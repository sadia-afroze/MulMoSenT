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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import BlipProcessor, BlipModel,  pipeline

label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}

class CustomBLIPDataloder(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["ReLabel"].tolist()
        self.image_path = self.data["Image"].tolist()
        self.samples = len(self.data)
        self.text = self.data["Text"].tolist()
        self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    def __len__(self):
        return self.samples
    def __getitem__(self,idx):
        image_path = self.image_path[idx]
        label = label_map[self.label[idx]]
        #image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(image_path)
        image = self.transform(image)
        text = self.text[idx]
        encoded_input = self.tokenizer(text=text, images=image, padding='max_length', truncation=True, max_length=14, return_tensors='pt')
        return {'pixel_values': image, 'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'image_path': self.image_path[idx],'label': torch.tensor(label,dtype=torch.long)}



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Print Device',device)


if __name__ == "__main__":
    data = pd.read_csv("test.csv")
    Text = data["Text"].tolist()
    Image_path = data["Image"].tolist()
    label = data["ReLabel"].tolist()
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    categories =["Neutral", "Positive", "Negative"]
    pred_labels = []
    actual_labels = []
    for i in range(len(Text)):
        text = Text[i]
        image = Image.open(Image_path[i])
        labelss = label[i]
        label_idx = label_map[labelss]
        cominedTextCategory = [f"{text},{category}" for category in categories]
        inputs = processor(text=cominedTextCategory, images=image, return_tensors="pt", padding='max_length',
                           truncation=True, max_length=30)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        best_image_prob = logits_per_image.softmax(dim=1).argmax().item()
        # best_text_prob = logits_per_text.softmax(dim=1).argmax().item()
        pred_labels.append(best_image_prob)
        actual_labels.append(label_idx)

    print("Accuracy: ", accuracy_score(actual_labels, pred_labels))
    print("Precision: ", precision_score(actual_labels, pred_labels, average='macro'))
    print("Recall: ", recall_score(actual_labels, pred_labels, average='macro'))
    print("F1 Score: ", f1_score(actual_labels, pred_labels, average='macro'))
    print("Classification Report: ", classification_report(actual_labels, pred_labels))
    print("Confusion Matrix: ", confusion_matrix(actual_labels, pred_labels))


