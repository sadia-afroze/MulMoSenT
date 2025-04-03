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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import ViTModel, ElectraModel, ElectraTokenizer, get_linear_schedule_with_warmup
import torchvision as tv
label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class CustomDataLoaderMMTC(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.image_path = self.data["Image"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = 224
        self.tokenizer = ElectraTokenizer.from_pretrained("csebuetnlp/banglabert")

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
        return {'image': image, 'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'image_path':self.image_path[idx], 'label': torch.tensor(label,dtype=torch.long), 'text': text}

class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.5, 0.5, 1.0], gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)  # Move alpha to the correct device here
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets)
        at = self.alpha[targets.data.view(-1).long()]  # Now alpha and targets are on the same device
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
class CustomImageClassification(torch.nn.Module):
    def __init__(self, num_labels=3):
        super(CustomImageClassification, self).__init__()
        self.hidden_size = 1000
        self.num_labels = 3
        self.model = tv.models.resnet50(pretrained=True)  #ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", return_dict=True)
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values):
        output = self.model(pixel_values)
        output = self.dropout(output)
        #print(output.shape)
        return output



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
        output = self.dropout(output.last_hidden_state[:, 0, :])
        return output


class CustoMultiModalImgaeTextClassification(torch.nn.Module):
    def __init__(self, num_labels=3, image_model=None, text_model=None):
        super(CustoMultiModalImgaeTextClassification, self).__init__()
        self.hidden_size = 1768
        self.num_labels = num_labels
        self.image_model = image_model
        self.text_model = text_model
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values, input_ids, attention_mask):
        image_output = self.image_model(pixel_values)
        text_output = self.text_model(input_ids, attention_mask)
        output = torch.cat((image_output, text_output), dim=1)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output




def TestMMTC():
    best_model_path = '../Model/Image-based-Sentiment/best_mmtc_model_withoutfreezResNeT50_BanglaBERT.pth'
    dataset = CustomDataLoaderMMTC("../Machine_Annotation_Test.csv")
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torch.load(best_model_path).to(device)
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
            text_list.extend(text_array)
            Image_path_list.extend(image_path)

    with open('../Model/MSA_FocalResNet50_BanglaBERT_losswithoutfreez_missclassified.csv', 'w') as f:
        f.write('Image_path,Text,Predicted_Label,Actual_Label\n')
        for i in range(len(predict_labels)):
            if predict_labels[i] != actual_labels[i]:
                f.write(f'{Image_path_list[i]},{text_list[i]},{predict_labels[i]},{actual_labels[i]}\n')
    acc = accuracy_score(actual_labels, predict_labels)
    print("Accuracy: ", acc)
    print(classification_report(actual_labels, predict_labels))
    print(confusion_matrix(actual_labels, predict_labels))
    return acc





def TrainMMTC():
    datasets = CustomDataLoaderMMTC("../SentimentImages/Machine_Annotation_Train.csv")
    dataloader = DataLoader(datasets, batch_size=64, shuffle=True)
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
    certiation_loss = FocalLoss() #torch.nn.CrossEntropyLoss()
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
            torch.save(best_model, "../Model/Image-based-Sentiment/best_mmtc_model_withoutfreezResNeT50_BanglaBERT.pth")




if __name__ == "__main__":
    TestMMTC()
    #TrainMMTC()
    #TrainTextModel()
    #TestTextModel()
    #TrainImageModel()
    #TestImageModel()