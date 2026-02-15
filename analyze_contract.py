import fitz
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter

# load model
model_path = "risk_clause_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# risk weights
risk_weights = {
    "Termination": 3,
    "Liability": 3,
    "Indemnification": 3,
    "Payment Terms": 2,
    "Warranty": 2,
    "Confidentiality": 1,
    "Force Majeure": 1,
    "Governing Law": 1,
    "Intellectual Property": 2
}


# extract pdf text
def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    text = ""
    for page in doc:
        text += page.get_text()

    return text


# split clauses
def split_clauses(text):

    clauses = text.split(".")
    clauses = [c.strip() for c in clauses if len(c.strip()) > 40]

    return clauses


# predict
def predict_clause(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    pred_id = torch.argmax(probs).item()

    category = model.config.id2label[pred_id]

    confidence = probs[0][pred_id].item()

    return category, confidence


# analyzer
def analyze_contract(pdf_path):

    print("\nAnalyzing contract...\n")

    text = extract_text_from_pdf(pdf_path)
    clauses = split_clauses(text)

    results = []
    total_risk = 0
    categories = []

    for i, clause in enumerate(clauses):

        category, confidence = predict_clause(clause)

        risk = risk_weights.get(category, 1)

        total_risk += risk
        categories.append(category)

        results.append({
            "Clause_Number": i+1,
            "Category": category,
            "Confidence": round(confidence, 3),
            "Risk_Score": risk,
            "Clause_Text": clause
        })


    # summary
    avg_risk = total_risk / len(clauses)

    if avg_risk >= 2.5:
        verdict = "HIGH RISK"
    elif avg_risk >= 1.5:
        verdict = "MEDIUM RISK"
    else:
        verdict = "LOW RISK"


    category_counts = Counter(categories)


    # clean output
    print("========== CONTRACT SUMMARY ==========\n")

    print("Total Clauses:", len(clauses))
    print("Average Risk Score:", round(avg_risk, 2))
    print("Final Verdict:", verdict)

    print("\nClause Category Distribution:")

    for cat, count in category_counts.items():
        print(f"{cat}: {count}")

    print("\nFull results saved in contract_analysis_results.csv")
    print("=====================================\n")


    # save csv
    df = pd.DataFrame(results)
    df.to_csv("contract_analysis_results.csv", index=False)


# run
analyze_contract("contract.pdf")
