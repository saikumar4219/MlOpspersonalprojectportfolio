from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load trained model
model_path = "./bert_log_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load Jenkins logs
def get_jenkins_logs():
    with open("logs.txt", "r") as file:
        return [line.strip() for line in file.readlines() if line.strip()]

# Predict log types
def classify_logs(logs):
    inputs = tokenizer(logs, padding=True, truncation=True, max_length=128, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.tolist()

# Run analysis
logs = get_jenkins_logs()
predictions = classify_logs(logs)

# Map predictions back to labels
label_map = {0: "Success", 1: "Error", 2: "Warning", 3: "Info"}
for log, pred in zip(logs, predictions):
    print(f"Log: {log}\nPrediction: {label_map[pred]}\n")
