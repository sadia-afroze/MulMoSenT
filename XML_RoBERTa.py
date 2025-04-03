import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import re
from torchvision import transforms, utils
import cv2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import XLMRobertaModel, XLMRobertaTokenizer,get_linear_schedule_with_warmup, AdamW, AutoConfig, AutoModel



label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}


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




class CustomDataLoaderLP(Dataset):
    def __init__(self,csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["ReLabel"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    def __len__(self):
        return self.samples

    def __getitem__(self,idx):
        text = self.sentences[idx]
        label = self.label[idx]
        #print(label)
        label = label_map[label]
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        return {'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'label': torch.tensor(label,dtype=torch.long)}

class XML_RoBERTa(torch.nn.Module):
    def __init__(self, num_labels=3):
        super(XML_RoBERTa, self).__init__()
        self.config = AutoConfig.from_pretrained("xlm-roberta-base", num_labels=3)
        self.config.return_dict = True
        self.model = AutoModel.from_pretrained("xlm-roberta-base", return_dict=True)
        self.hidden_size = 768
        self.linear = torch.nn.Linear(self.hidden_size, self.config.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state[:, 0, :]
        output = self.linear(output)
        output = self.softmax(output)
        return output


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ValidateXMLRoBERTa():
    datasets = CustomDataLoaderLP("./test.csv")
    dataloader = DataLoader(datasets, batch_size=256, shuffle=True)
    Eval_model = torch.load('./Ablation_Model/BenEngBest_XML_RoBERTa_model_Focal.pt').to(device)
    #Eval_model = temp_model
    predict_labels = []
    actual_labels = []

    with torch.no_grad():
        Eval_model.eval()
        for i, data in enumerate(dataloader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['label']
            output = Eval_model(input_ids, attention_mask)
            output = output.argmax(dim=1).cpu().numpy()
            label = label.cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(label)
    accuracy = accuracy_score(actual_labels, predict_labels)
    print(f"Accuracy: {accuracy}")
    print(f"{classification_report(actual_labels, predict_labels)}")
    print(f"Confusion Matrix:\n {confusion_matrix(actual_labels, predict_labels)}")
    return accuracy



def TrainXMLRoBERTa():
    datasets = CustomDataLoaderLP("./train.csv")
    dataloader = DataLoader(datasets, batch_size=16, shuffle=True)
    model = XML_RoBERTa()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)*10)
    class_counts = [5713, 4027, 6003]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
    criterion = criterion #torch.nn.CrossEntropyLoss()
    model.train()
    best_accuracy = 0
    best_model = None
    for epoch in range(20):
        total_loss = 0
        total_accuracy = 0
        predict_labels = []
        actual_labels = []
        for i, data in enumerate(dataloader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['label']
            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            label = torch.nn.functional.one_hot(label, num_classes=3).to(device)
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()
            output = output.argmax(dim=1).cpu().numpy()
            label = label.argmax(dim=1).cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(label)
            total_loss += loss.item()/data['input_ids'].shape[0]
        scheduler.step()
        accuracy = ValidateXMLRoBERTa(model)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            torch.save(best_model, "./Ablation_Model/BenEngBest_XML_RoBERTa_model_Focal.pt")

def DataValidityCheck(path1, path2):
    data1 = pd.read_csv(path1, encoding='utf-8')
    text1 = data1["Text"].tolist()
    label1 = data1["Label"].tolist()
    data2 = pd.read_csv(path2, encoding='utf-8')
    text2 = data2["Text"].tolist()
    label2 = data2["Label"].tolist()

    with open("/media/tigerit/ssd/Multimodal-Transformer/ml-projects-main/sentiment_analysis_bangla/Data_combined/CyberToxic/Train.csv", "a+", encoding='utf-8') as f:
        f.write("Text,Label\n")
        for i in range(len(text1)):
            f.write(f"{text1[i]},{label1[i]}\n")

    with open("/media/tigerit/ssd/Multimodal-Transformer/ml-projects-main/sentiment_analysis_bangla/Data_combined/CyberToxic/Train.csv", "a+", encoding='utf-8') as f:
        for i in range(len(text2)):
            f.write(f"{text2[i]},{label2[i]}\n")

if __name__ == "__main__":
    ValidateXMLRoBERTa()
    #TrainXMLRoBERTa()

    #path = './Model/BenHiBest_XML_RoBERTa_model.pt'
    #ValidateXMLRoBERTa(path)

