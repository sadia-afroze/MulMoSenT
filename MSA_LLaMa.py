from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import torch
import cv2
import math
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import ViTModel, get_linear_schedule_with_warmup
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
label_map = {"Positive": 1, "Negative": 2, "Neutral": 0}


# Define Dataset
class CustomDataLoaderMMTC(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data["ReLabel"].tolist()
        self.image_paths = self.data["Image"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.image_size = 224
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # Image preprocessing
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = self.transform(image)

        # Text tokenization
        text = self.sentences[idx]
        label = label_map[self.labels[idx]]
        encoded_input = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=30, return_tensors="pt"
        )
        return {
            "image": image,
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Multi-modal Model with LLaMA and Vision Encoder
class MultiModalModel(nn.Module):
    def __init__(self, num_labels=3):
        super(MultiModalModel, self).__init__()
        self.text_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",return_dict=True,output_hidden_states=True)
        self.embedding_size = 2048
        self.classifier = nn.Linear(self.embedding_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # Process text and image inputs
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # Extract logits for the last token (causal models predict one token at a time)
        hidden_states = output.hidden_states[-1]  # shape: [batch_size, sequence_length, hidden_size]
        # Pool the hidden states to get CLS-like embedding
        cls_embedding = torch.mean(hidden_states, dim=1)  # Avera
        #print(cls_embedding.shape)
        logits = self.classifier(cls_embedding)
        return self.softmax(logits)

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


def test_mmtc(model):
    # Load test dataset
    dataset = CustomDataLoaderMMTC("./test.csv")
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False)

    # Initialize model and load trained weights
    model = model #MultiModalModel(num_labels=3).to(device)
    model = model.to(device)
    #model_dict = torch.load("./Baseline-Models/LLaMa.pth")
    #model.load_state_dict(model_dict, strict=False)
    #model = temp_model
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # Forward pass
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=list(label_map.keys()))
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Print results
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    return acc


# Training function with QLoRA
def train_mmtc():
    dataset = CustomDataLoaderMMTC("./train.csv")
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    # Model and optimizer setup
    model = MultiModalModel(num_labels=3).to(device)

    # QLoRA setup
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.2,
        target_modules=["q_proj", "v_proj"],
    )
    model.text_model = get_peft_model(model.text_model, peft_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=20 * len(dataloader))

    class_counts = [5713, 4027, 6003]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')

    criterion = nn.CrossEntropyLoss()
    model.train()
    globalLoss = 1000000000000
    globalacc = 0
    for epoch in range(20):
        epoch_loss = 0
        all_preds = []
        all_labels = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = test_mmtc(model)
        if globalacc < acc:
            globalacc = acc
            torch.save(model.state_dict(), "./Baseline-Models/LLaMa.pth")
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")

# Main entry point
if __name__ == "__main__":
    #test_mmtc()
    train_mmtc()

