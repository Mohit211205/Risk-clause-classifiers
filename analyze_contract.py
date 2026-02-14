import fitz  # PyMuPDF
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================
# LOAD TRAINED MODEL
# ============================

model_path = "risk_clause_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# ============================
# RISK WEIGHTS
# ============================

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

# ============================
# EXTRACT TEXT FROM PDF
# ============================

def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


# ============================
# SPLIT INTO CLAUSES
# ============================

def split_clauses(text):

    clauses = text.split(".")
    clauses = [c.strip() for c in clauses if len(c.strip()) > 40]

    return clauses


# ============================
# PREDICT CLAUSE CATEGORY
# ============================

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

    predicted_class_id = torch.argmax(probs).item()

    category = model.config.id2label[predicted_class_id]

    confidence = probs[0][predicted_class_id].item()

    return category, confidence


# ============================
# MAIN ANALYZER
# ============================

def analyze_contract(pdf_path):

    text = extract_text_from_pdf(pdf_path)

    clauses = split_clauses(text)

    print("\nContract Analysis Results:\n")

    results = []

    total_risk_score = 0

    for i, clause in enumerate(clauses):

        category, confidence = predict_clause(clause)

        risk_score = risk_weights.get(category, 1)

        total_risk_score += risk_score

        print(f"Clause {i+1}: {category} (confidence: {confidence:.2f})")

        results.append({
            "Clause_Number": i+1,
            "Clause_Text": clause,
            "Category": category,
            "Confidence": round(confidence, 3),
            "Risk_Score": risk_score
        })


    # ============================
    # FINAL VERDICT LOGIC
    # ============================

    avg_risk = total_risk_score / len(clauses)

    if avg_risk >= 2.5:
        verdict = "HIGH RISK"
    elif avg_risk >= 1.5:
        verdict = "MEDIUM RISK"
    else:
        verdict = "LOW RISK"


    print("\n===========================")
    print("TOTAL CLAUSES:", len(clauses))
    print("TOTAL RISK SCORE:", total_risk_score)
    print("AVERAGE RISK:", round(avg_risk, 2))
    print("FINAL VERDICT:", verdict)
    print("===========================\n")


    # ============================
    # SAVE CSV REPORT
    # ============================

    df = pd.DataFrame(results)

    df.to_csv("contract_analysis_results.csv", index=False)

    print("Saved report → contract_analysis_results.csv")


# ============================
# RUN ANALYSIS
# ============================

analyze_contract("contract.pdf")
