<<<<<<< HEAD
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
=======
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

>>>>>>> 9f485b5 (merged github and local repo)
model_path = "risk_clause_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

<<<<<<< HEAD
# Label mapping
id2label = model.config.id2label

def predict_clause(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return id2label[prediction]


# Test
text = "You may cancel your contract within 10 days."

result = predict_clause(text)

print("Prediction:", result)
=======
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
>>>>>>> 9f485b5 (merged github and local repo)
