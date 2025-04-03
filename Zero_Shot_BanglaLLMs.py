import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from transformers import pipeline

# Load data
data = pd.read_csv('test-mmtc-6.csv')
Text = data['Text'].tolist()
label = data['Label'].tolist()

# Define label map
label_map = {"Business": 0, "Crime": 1, "Entertainment": 2, "Environment": 3, "Science-Tech": 4, "Others": 5}

# Initialize lists for predictions and actual labels
pred_labels = []
actual_labels = []

# Initialize classifier
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline('zero-shot-classification', model='BanglaLLM/bangla-llama-7b-base-v0.1', device=device)
candidate_labels = ["Business", "Crime", "Entertainment", "Environment", "Science-Tech", "Others"]

# Process each text
corrected = 0
for i, text in enumerate(Text):
    labelss = label[i]
    label_idx = label_map[labelss]
    result = classifier(text, candidate_labels)

    best_label = result['labels'][0]
    if best_label in label_map:
        best_prob = label_map[best_label]
        pred_labels.append(best_prob)
        actual_labels.append(label_idx)

        if best_prob == label_idx:
            corrected += 1
    else:
        print(f"Warning: Label {best_label} not found in label_map")

    print(f'Corrected: {corrected}/{i + 1}')

# Evaluate the results
print("Accuracy: ", accuracy_score(actual_labels, pred_labels))
print("Precision: ", precision_score(actual_labels, pred_labels, average='macro'))
print("Recall: ", recall_score(actual_labels, pred_labels, average='macro'))
print("F1 Score: ", f1_score(actual_labels, pred_labels, average='macro'))
print("Classification Report: \n", classification_report(actual_labels, pred_labels))
print("Confusion Matrix: \n", confusion_matrix(actual_labels, pred_labels))
