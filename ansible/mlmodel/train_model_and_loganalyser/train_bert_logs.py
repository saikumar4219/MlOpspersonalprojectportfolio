import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset (logs labeled as "Error", "Warning", "Success", etc.)
df = pd.read_csv("jenkins_logs.csv")

# Expand dataset if multiple log messages exist for a single label
expanded_data = []
for index, row in df.iterrows():
    messages = row["log_message"].split("|")  # Assuming messages are separated by "|"
    for msg in messages:
        expanded_data.append({"log_message": msg.strip(), "label": row["label"]})

df_expanded = pd.DataFrame(expanded_data)

# Encode labels to numbers
label_encoder = LabelEncoder()
df_expanded["label"] = label_encoder.fit_transform(df_expanded["label"])

# Tokenize logs
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encodings = tokenizer(list(df_expanded["log_message"]), truncation=True, padding=True, max_length=128)

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict({"text": list(df_expanded["log_message"]), "label": list(df_expanded["label"])})

# Split into training and test sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Load Pretrained BERT model for classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./bert_log_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save trained model and tokenizer
model.save_pretrained("./bert_log_model")
tokenizer.save_pretrained("./bert_log_model")

print("Model training complete. Saved in ./bert_log_model")
