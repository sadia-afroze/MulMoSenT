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

from transformers import BridgeTowerModel, BridgeTowerProcessor, AdamW, get_linear_schedule_with_warmup, AutoConfig

label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}


if __name__ == '__main__':
    data = pd.read_csv("./test.csv")
    Text = data["Text"].tolist()
    Image_path = data["Image"].tolist()
    label = data["ReLabel"].tolist()
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
    model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
    categories = ["Neutral", "Positive", "Negative"]
    pred_labels = []
    actual_labels = []
    for i in range(len(Text)):
        text = Text[i]
        image = Image.open(Image_path[i])
        image = image
        labelss = label[i]
        label_idx = label_map[labelss]
        cominedTextCategory = [f"{text},{category}" for category in categories]
        cominedTextCategory = cominedTextCategory

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
