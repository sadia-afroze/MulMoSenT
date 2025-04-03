import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import cv2
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, ViTModel, get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}


# Dataset class
class CustomDataLoaderMMTC(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data["ReLabel"].tolist()
        self.image_paths = self.data["Image"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.image_size = 224
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.tokenizer = AutoTokenizer.from_pretrained("proxectonos/Carballo-bloom-1.3B")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = self.transform(image)

        text = self.sentences[idx]
        label = label_map[self.labels[idx]]
        encoded_input = self.tokenizer(text, padding="max_length", truncation=True, max_length=30, return_tensors="pt")

        return {
            "image": image,
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Multi-modal Model
class MultiModalModel(nn.Module):
    def __init__(self, num_labels=3):
        super(MultiModalModel, self).__init__()
        self.text_model = AutoModelForCausalLM.from_pretrained("proxectonos/Carballo-bloom-1.3B", return_dict=True,
                                                               output_hidden_states=True)
        self.embedding_size = 2048
        self.classifier = nn.Linear(self.embedding_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output.hidden_states[-1]
        cls_embedding = torch.mean(hidden_states, dim=1)
        logits = self.classifier(cls_embedding)
        return self.softmax(logits)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            targets = targets.long()
            alpha = self.alpha[targets].view(-1, 1)
            focal_loss = alpha * focal_loss

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


# Testing function
def test_mmtc(model, dataset_path="./test.csv"):
    dataset = CustomDataLoaderMMTC(dataset_path)
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False)
    model.to(device).eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
            batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(
        f"Test Accuracy: {acc:.4f}\nClassification Report:\n{classification_report(all_labels, all_preds, target_names=list(label_map.keys()))}\nConfusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")
    return acc


# Training function
def train_mmtc():
    dataset = CustomDataLoaderMMTC("./train.csv")
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    model = MultiModalModel(num_labels=3).to(device)

    #peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.2,
    #                         target_modules=["q_proj", "v_proj"])

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.2,
        target_modules=["query_key_value"],  # Use "query_key_value" for BLOOM
    )

    model.text_model = get_peft_model(model.text_model, peft_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500,
                                                num_training_steps=20 * len(dataloader))

    class_counts = [5713, 4027, 6003]
    class_weights = (1. / torch.tensor(class_counts, dtype=torch.float)).to(device)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss()  #FocalLoss(alpha=class_weights, gamma=2, reduction='mean')

    best_acc = 0
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
            batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        acc = test_mmtc(model)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./Baseline-Models/LLaMa.pth")
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train_mmtc()
