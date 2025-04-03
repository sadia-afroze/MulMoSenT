import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import re

#from torch.xpu import device
from torchvision import transforms, utils
import cv2
from torch.cuda.amp import GradScaler, autocast
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import AlignModel, AlignProcessor, AdamW, get_linear_schedule_with_warmup, AutoConfig

label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageTextDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["ReLabel"].tolist()
        self.image_path = self.data["Image"].tolist()
        self.text = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = 224
        self.tokenizer = AlignProcessor.from_pretrained('kakaobrain/align-base')

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
        encoded_input = self.tokenizer(images=image, text=text, padding='max_length', truncation=True, max_length=14,
                                   return_tensors='pt')
        return {
        'pixel_values': encoded_input['pixel_values'].squeeze(0),
        'input_ids': encoded_input['input_ids'].squeeze(0),
        'attention_mask': encoded_input['attention_mask'].squeeze(0),
        'label': torch.tensor(label, dtype=torch.long)
        }

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


class Align(torch.nn.Module):
    def __init__(self, num_labels=3):
        super(Align, self).__init__()
        self.config = AutoConfig.from_pretrained("kakaobrain/align-base", num_labels=3)
        self.config.return_dict = True
        self.model = AlignModel.from_pretrained("kakaobrain/align-base", return_dict=True)
        self.text_embedding_size = 640
        self.image_embedding_size = 640
        self.input_size =  50271
        self.fc = torch.nn.Linear(self.input_size+1, self.text_embedding_size)
        self.linear = torch.nn.Linear(self.text_embedding_size+self.image_embedding_size, self.config.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values, attention_mask,input_ids):
        output = self.model(pixel_values=pixel_values, attention_mask=attention_mask, input_ids=input_ids)
        #For CLIP and BLIP
        image_embeds = output["image_embeds"]
        text_embeds = output["text_embeds"]
        #print(image_embeds.shape)
        #print(text_embeds.shape)
        catFeatures = torch.cat((image_embeds, text_embeds), dim=1)
        output = self.linear(catFeatures)
        output = self.softmax(output)
        return output


def TestBMMTC_Align():
    model = Align()
    model = torch.load("./Baseline-Models/ALIGN_Cross.pth.pt")
    model.to(device)
    model.eval()
    datasets = ImageTextDataset("./test.csv")
    dataloader = DataLoader(datasets, batch_size=10, shuffle=True)
    predictions = []
    actual = []
    image_path_list = []
    for batch in dataloader:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label']
        #image_path = batch['image_path']
        output = model(pixel_values=pixel_values, attention_mask=attention_mask,input_ids=input_ids)
        output = output.argmax(dim=1).cpu().numpy()
        #image_path_list.extend(image_path)
        predictions.extend(output)
        label = label.numpy()
        actual.extend(label)


    #Write csv file for image path and predicted label, actual label for misclassified images
    # with open('./mmbtc-6-model/best_model_CLIP_missclassified.csv', 'w') as f:
    #     f.write('Image_path, Predicted_Label, Actual_Label\n')
    #     for i in range(len(predictions)):
    #         if predictions[i] != actual[i]:
    #             f.write(f'{image_path_list[i]}, {predictions[i]}, {actual[i]}\n')

    acc = accuracy_score(actual, predictions)
    print("Accuracy:", acc)
    print("Precision:", precision_score(actual, predictions, average='macro'))
    print("Recall:", recall_score(actual, predictions, average='macro'))
    print("F1 Score:", f1_score(actual, predictions, average='macro'))
    print("Classification Report:", classification_report(actual, predictions))
    print("Confusion Matrix:", confusion_matrix(actual, predictions))
    return acc



def TrainBMMTC_Align():

    model = Align()
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    certian_loss = torch.nn.CrossEntropyLoss()
    datasets = ImageTextDataset("./train.csv")
    dataloader = DataLoader(datasets, batch_size=8, shuffle=True)
    epochs = 15
    best_model = None
    global_loss = 10000000000
    bestACC = 0.0
    class_counts = [5713, 4027, 6003]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
    for epoch in range(epochs):
        print("Epoch:", epoch)
        predictions = []
        actual = []
        epoch_loss = 0
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label']
            optimizer.zero_grad()
            output = model(pixel_values=pixel_values, attention_mask=attention_mask,input_ids=input_ids)
            label = torch.nn.functional.one_hot(label, num_classes=3).to(device)
            loss = certian_loss(output, label.float())
            loss.backward()
            optimizer.step()
            output = output.argmax(dim=1).cpu().numpy()
            predictions.extend(output)
            label = label.argmax(dim=1).cpu().numpy()
            actual.extend(label)
            epoch_loss += loss.item()/batch["label"].shape[0]
        scheduler.step()
        accuracyscore = accuracy_score(actual, predictions)
        print("Accuracy:", accuracyscore)
        print("Loss:", epoch_loss)
        #acc = TestBMMTC_Align(model)
        print(f'Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {accuracyscore}')
        if epoch_loss < global_loss:
            global_loss = epoch_loss
            best_model = model
            torch.save(best_model, "./Baseline-Models/ALIGN_Cross.pth.pt")


if __name__ == "__main__":
    #TrainBMMTC_Align()
    TestBMMTC_Align()

