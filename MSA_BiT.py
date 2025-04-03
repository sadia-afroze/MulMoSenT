import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import re
from torchvision import transforms, utils
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import BitModel, AdamW, get_linear_schedule_with_warmup, AutoConfig, BitConfig

label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}



class CustomImageDataLoader(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["ReLabel"].tolist()
        self.image_path = self.data["Image"].tolist()
        self.samples = len(self.data)
        self.transform =  transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.image_size = 224

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("Converted to RGB")

        image = self.transform(image)
        label = self.label[idx]
        label = label_map[label]
        return {'image': image, 'label': torch.tensor(label,dtype=torch.long)}


class CustomBiTImageClassification(torch.nn.Module):
    def __init__(self, num_labels=3):
        super(CustomBiTImageClassification, self).__init__()
        self.config = AutoConfig.from_pretrained("google/bit-50", num_labels=3)
        self.config.return_dict = True
        self.model = BitModel.from_pretrained("google/bit-50", return_dict=True)
        #print(self.model)
        #2048 x 4 x 4
        self.flatten_dim = 32768
        self.hidden_size = 2048
        self.num_labels = 3
        self.fc = torch.nn.Linear(self.flatten_dim, self.hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values):
        output = self.model(pixel_values=pixel_values)
        #Extract 2048 from the last hidden state
        output = output.last_hidden_state
        output = output.mean(dim=[2,3])
        #print(output.shape)
        output = self.dropout(output)
        #output = self.fc(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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


def TestBiTModel():
    dataset = CustomImageDataLoader("./test.csv")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    num_labels = 3
    model = CustomBiTImageClassification()
    model = model.to(device)
    model.load_state_dict(torch.load("./Ablation_Model/best_model_BiT_image_Focal.pt"))
    model.eval()
    predict_labels = []
    actual_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"]
            outputs = model(images)
            outputs = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            predict_labels.extend(outputs)
            actual_labels.extend(labels)

    print(classification_report(actual_labels, predict_labels))
    print(confusion_matrix(actual_labels, predict_labels))
    acc = accuracy_score(actual_labels, predict_labels)
    print("Accuracy: ", acc)
    return acc



def TrainBiTModel():
    datasets = CustomImageDataLoader("./train.csv")
    dataloader = DataLoader(datasets, batch_size=16, shuffle=True)
    num_labels = 3
    model = CustomBiTImageClassification()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 10)

    class_counts = [5713, 4027, 6003]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')

    certiation_loss = criterion #torch.nn.CrossEntropyLoss()
    epochs = 30
    best_accuracy = 0
    best_loss = 100000000
    best_model = None
    for epoch in range(epochs):
        running_loss = 0.0
        predict_labels = []
        actual_labels = []
        for i, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"]
            optimizer.zero_grad()
            labels = torch.nn.functional.one_hot(labels, num_classes=num_labels).to(device)
            outputs = model(images)
            #print(outputs)
            #print(labels.float())
            loss = certiation_loss(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss +=  loss.item()/batch["image"].shape[0]
            outputs = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.argmax(dim=1).cpu().numpy()
            predict_labels.extend(outputs)
            actual_labels.extend(labels)
        scheduler.step()
        print("Epoch: ", epoch, "Loss: ", running_loss)
        acc = TestBiTModel(model)
        if best_accuracy < acc:
            best_accuracy = acc
            best_model = model
            torch.save(best_model.state_dict(), "./Ablation_Model/best_model_BiT_image_Focal.pt")




if __name__ == "__main__":
    #TrainBiTModel()
    TestBiTModel()

