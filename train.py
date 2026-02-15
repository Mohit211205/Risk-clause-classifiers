import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ------------------------
# LOAD DATA
# ------------------------

df = pd.read_csv("risk_clause_labelled.csv")

df = df.dropna()
df = df.drop_duplicates()

labels = df["Category"].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["labels"] = df["Category"].map(label2id)
dataset = Dataset.from_pandas(df[["Clause_Text", "labels"]])
dataset = dataset.train_test_split(test_size=0.2)

# ------------------------
# TOKENIZER
# ------------------------

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["Clause_Text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.remove_columns(["Clause_Text"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# ------------------------
# MODEL
# ------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# ------------------------
# METRICS
# ------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# ------------------------
# TRAINING ARGS
# ------------------------

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    load_best_model_at_end=True
)

# ------------------------
# TRAINER
# ------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# ------------------------
# TRAIN
# ------------------------

trainer.train()

# ------------------------
# EVALUATE
# ------------------------

results = trainer.evaluate()
print("\nEvaluation Results:")
print(results)

# Detailed Report
predictions = trainer.predict(dataset["test"])
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=labels))

# ------------------------
# SAVE MODEL
# ------------------------

model.save_pretrained("risk_clause_model")
tokenizer.save_pretrained("risk_clause_model")

print("\nModel saved in folder: risk_clause_model")
