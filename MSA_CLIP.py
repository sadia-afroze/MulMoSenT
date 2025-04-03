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

from transformers import CLIPModel, CLIPProcessor, AdamW, get_linear_schedule_with_warmup, AutoConfig, BlipProcessor, BlipModel, Blip2Model, Blip2Processor

label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}

class CustomCLIPDataloder(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["ReLabel"].tolist()
        self.image_path = self.data["Image"].tolist()
        self.samples = len(self.data)
        self.text = self.data["Text"].tolist()
        self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        #self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        #self.tokenizer = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",load_in_8bit=True)
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    def __len__(self):
        return self.samples
    def __getitem__(self,idx):
        image_path = self.image_path[idx]
        label = label_map[self.label[idx]]
        image = Image.open(image_path)
        #check the image dimension
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("Converted to RGB")
        image = self.transform(image)
        text = self.text[idx]
        encoded_input = self.tokenizer(text=text, images=image, padding='max_length', truncation=True, max_length=14, return_tensors='pt')
        return {'pixel_values': image, 'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'image_path': self.image_path[idx],'label': torch.tensor(label,dtype=torch.long)}

class CustomCLIP(torch.nn.Module):
    def __init__(self, num_labels=3):
        super(CustomCLIP, self).__init__()
        self.config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32", num_labels=3)
        self.config.return_dict = True
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", return_dict=True)
        self.text_embedding_size = 512
        self.image_embedding_size = 512
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
        image_embeds = image_embeds.to(dtype=torch.float32)
        text_embeds = text_embeds.to(dtype=torch.float32)
        # print(image_embeds.shape)
        # print(text_embeds.shape)
        fused = torch.cat((image_embeds, text_embeds), dim=1)
        fused = self.dropout(fused)
        res = self.linear(fused)
        res = self.softmax(res)
        return res


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Print Device',device)


def TrainBMMTC_CLIP():
    model = CustomCLIP()
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    certian_loss = torch.nn.CrossEntropyLoss()
    datasets = CustomCLIPDataloder("./train.csv")
    dataloader = DataLoader(datasets, batch_size=12, shuffle=True)
    epochs = 15
    best_model = None
    global_loss = 10000000000
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
        if epoch_loss < global_loss:
            global_loss = epoch_loss
            best_model = model
    torch.save(best_model, "./mmbtc-6-model/best_model_CLIP.pt")


def TestBMMTC_CLIP():
    model = torch.load("./mmbtc-6-model/best_model_CLIP.pt")
    model.to(device)
    model.eval()
    datasets = CustomCLIPDataloder("./test.csv")
    dataloader = DataLoader(datasets, batch_size=32, shuffle=True)
    predictions = []
    actual = []
    image_path_list = []
    for batch in dataloader:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label']
        image_path = batch['image_path']
        output = model(pixel_values=pixel_values, attention_mask=attention_mask,input_ids=input_ids)
        output = output.argmax(dim=1).cpu().numpy()
        image_path_list.extend(image_path)
        predictions.extend(output)
        label = label.numpy()
        actual.extend(label)


    #Write csv file for image path and predicted label, actual label for misclassified images
    # with open('./mmbtc-6-model/best_model_CLIP_missclassified.csv', 'w') as f:
    #     f.write('Image_path, Predicted_Label, Actual_Label\n')
    #     for i in range(len(predictions)):
    #         if predictions[i] != actual[i]:
    #             f.write(f'{image_path_list[i]}, {predictions[i]}, {actual[i]}\n')


    print("Accuracy:", accuracy_score(actual, predictions))
    print("Precision:", precision_score(actual, predictions, average='macro'))
    print("Recall:", recall_score(actual, predictions, average='macro'))
    print("F1 Score:", f1_score(actual, predictions, average='macro'))
    print("Classification Report:", classification_report(actual, predictions))
    print("Confusion Matrix:", confusion_matrix(actual, predictions))




if __name__ == "__main__":
    TrainBMMTC_CLIP()
