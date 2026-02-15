from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

model_path = "risk_clause_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

id2label = model.config.id2label


@app.get("/")
def home():
    return {"message": "Risk Clause Classifier API running"}


@app.post("/predict")
def predict(text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return {
        "clause": text,
        "prediction": id2label[prediction]
    }
