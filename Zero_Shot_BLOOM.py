import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from transformers.models.swiftformer.convert_swiftformer_original_to_hf import device

# Your Hugging Face token (if required)
token = "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"

# Initialize the tokenizer and model
model_name = "proxectonos/Carballo-bloom-1.3B" #"google/mt5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=token)

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Define candidate labels
candidate_labels = ["Neutral", "Positive", "Negative"]

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
print("Accuracy: ", accuracy_score(actual_labels_idx, pred_labels_idx))
print("Precision: ", precision_score(actual_labels_idx, pred_labels_idx, average='macro'))
print("Recall: ", recall_score(actual_labels_idx, pred_labels_idx, average='macro'))
print("F1 Score: ", f1_score(actual_labels_idx, pred_labels_idx, average='macro'))
print("Classification Report: \n", classification_report(actual_labels_idx, pred_labels_idx))
print("Confusion Matrix: \n", confusion_matrix(actual_labels_idx, pred_labels_idx))
