import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import re
from torchvision import transforms, utils
import cv2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import XLMRobertaModel, XLMRobertaTokenizer,get_linear_schedule_with_warmup, AdamW, AutoConfig, AutoModel, BertModel, BertTokenizer, BertConfig



label_map = {"Business": 0, "Crime": 1, "Entertainment": 2, "Environment": 3, "Science-Tech": 4, "Others": 5}



class CustomDataLoaderLP(Dataset):
    def __init__(self,csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.tokenizer = BertTokenizer.from_pretrained('google/muril-large-cased')

    def __len__(self):
        return self.samples

    def __getitem__(self,idx):
        text = self.sentences[idx]
        label = self.label[idx]
        #print(label)
        label = label_map[label]
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=14, return_tensors='pt')
        return {'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'label': torch.tensor(label,dtype=torch.long)}



class MuRIL(torch.nn.Module):
    def __init__(self, num_labels=6):
        super(MuRIL, self).__init__()
        self.config = AutoConfig.from_pretrained("google/muril-large-cased", num_labels=6)
        self.config.return_dict = True
        self.model = BertModel.from_pretrained("google/muril-large-cased", return_dict=True)
        self.hidden_size = 1024
        self.linear = torch.nn.Linear(self.hidden_size, self.config.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(output.last_hidden_state[:, 0, :])
        output = self.linear(output)
        output = self.softmax(output)
        return output


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ValidateMuRIL(temp_model):
    datasets = CustomDataLoaderLP("test-mmtc-6.csv")
    dataloader = DataLoader(datasets, batch_size=256, shuffle=True)
    #Eval_model = torch.load(model_path).to(device)
    Eval_model = temp_model
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
            label = label.numpy()
            predict_labels.extend(output)
            actual_labels.extend(label)
    accuracy = accuracy_score(actual_labels, predict_labels)
    print(f"Accuracy: {accuracy}")
    print(f"{classification_report(actual_labels, predict_labels)}")
    print(f"Confusion Matrix:\n {confusion_matrix(actual_labels, predict_labels)}")
    return accuracy



def TrainMuRIL():
    datasets = CustomDataLoaderLP("train-mmtc-6.csv")
    dataloader = DataLoader(datasets, batch_size=16, shuffle=True)
    model = MuRIL()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)*10)
    criterion = torch.nn.CrossEntropyLoss()
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
            label = torch.nn.functional.one_hot(label, num_classes=6).to(device)
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            output = output.argmax(dim=1).cpu().numpy()
            label = label.argmax(dim=1).cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(label)
            total_loss += loss.item()/data['input_ids'].shape[0]

        accuracy = ValidateMuRIL(model)
        print(f".......................{epoch}.......................\n")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    torch.save(best_model, "./mmbtc-6-model/google-muril-large-cased.pt")

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

    TrainMuRIL()

    #path = './Model/BenHiBest_XML_RoBERTa_model.pt'
    #ValidateXMLRoBERTa(path)

