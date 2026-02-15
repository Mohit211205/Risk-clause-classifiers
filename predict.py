from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "risk_clause_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_clause(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class_id = torch.argmax(probs).item()

    return model.config.id2label[predicted_class_id]

# test
clause = "Either party may terminate this agreement with written notice."

print("Clause:", clause)
print("Prediction:", predict_clause(clause))
