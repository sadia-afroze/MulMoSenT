import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, LlamaForCausalLM, LlamaTokenizer
from torch.cuda.amp import autocast
from transformers.models.swiftformer.convert_swiftformer_original_to_hf import device

# Your Hugging Face token
token = "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"

# Initialize the tokenizer and model
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForSequenceClassification.from_pretrained(model_name, load_in_8bit=True)

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Define candidate labels
candidate_labels =["Neutral", "Positive", "Negative"]

# Load your data (replace 'test-mmtc-6.csv' with your actual file)
data = pd.read_csv('./test.csv')  # Assuming your CSV file has 'Text' and 'Label' columns
texts = data['Text'].tolist()
actual_labels = data['ReLabel'].tolist()

# Initialize lists for predictions
pred_labels = []

# Perform classification
for i, text in enumerate(texts):
    result = classifier(text, candidate_labels)
    predicted_label = result['labels'][0]
    pred_labels.append(predicted_label)

    if (i + 1) % 10 == 0:
        print(f'Processed {i + 1}/{len(texts)} texts.')

# Map labels to indices
label_map = {label: i for i, label in enumerate(candidate_labels)}
pred_labels_idx = [label_map.get(label, -1) for label in pred_labels]
actual_labels_idx = [label_map.get(label, -1) for label in actual_labels]

# Evaluate the results using sklearn metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

print("Accuracy: ", accuracy_score(actual_labels_idx, pred_labels_idx))
print("Precision: ", precision_score(actual_labels_idx, pred_labels_idx, average='macro'))
print("Recall: ", recall_score(actual_labels_idx, pred_labels_idx, average='macro'))
print("F1 Score: ", f1_score(actual_labels_idx, pred_labels_idx, average='macro'))
print("Classification Report: \n", classification_report(actual_labels_idx, pred_labels_idx))
print("Confusion Matrix: \n", confusion_matrix(actual_labels_idx, pred_labels_idx))
