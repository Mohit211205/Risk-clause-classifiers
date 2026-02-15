from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi import BackgroundTasks
import fitz
import torch
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_path = "risk_clause_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

id2label = model.config.id2label


# -------------------------
# GLOBAL PROGRESS TRACKER
# -------------------------

progress_status = {
    "status": "idle",
    "progress": 0,
    "current_clause": 0,
    "total_clauses": 0,
    "estimated_seconds_remaining": 0
}


# -------------------------
# EXTRACT TEXT
# -------------------------

def extract_text_from_pdf(pdf_bytes):

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()

    return text


# -------------------------
# SPLIT CLAUSES
# -------------------------

def split_clauses(text):

    clauses = text.split(".")

    clauses = [
        c.strip()
        for c in clauses
        if len(c.strip()) > 40
    ]

    return clauses


# -------------------------
# PREDICT WITH CONFIDENCE
# -------------------------

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

    probs = torch.softmax(outputs.logits, dim=1)

    confidence, prediction = torch.max(probs, dim=1)

    category = id2label[prediction.item()]

    confidence_score = confidence.item()

    # Risk score logic
    if category in ["Indemnification", "Liability", "Termination"]:
        risk_score = 3
    elif category in ["Payment Terms", "Warranty"]:
        risk_score = 2
    else:
        risk_score = 1

    return category, confidence_score, risk_score


# -------------------------
# ANALYZE + SAVE CSV + PROGRESS
# -------------------------

@app.post("/analyze")
async def analyze_contract(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    pdf_bytes = await file.read()

    # start processing in background
    background_tasks.add_task(process_contract, pdf_bytes)

    return {
        "message": "Processing started"
    }

def process_contract(pdf_bytes):

    global progress_status

    progress_status["status"] = "processing"
    progress_status["progress"] = 0
    progress_status["current_clause"] = 0
    progress_status["total_clauses"] = 0
    progress_status["estimated_seconds_remaining"] = 0

    start_time = time.time()

    text = extract_text_from_pdf(pdf_bytes)

    clauses = split_clauses(text)

    total = len(clauses)

    progress_status["total_clauses"] = total

    results = []

    for i, clause in enumerate(clauses, 1):

        category, confidence, risk_score = predict_clause(clause)

        results.append({
            "Clause_Number": i,
            "Category": category,
            "Confidence": round(confidence, 3),
            "Risk_Score": risk_score,
            "Clause_Text": clause
        })

        progress = int((i / total) * 100)

        elapsed = time.time() - start_time

        estimated_total = (elapsed / i) * total
        remaining = int(estimated_total - elapsed)

        progress_status["progress"] = progress
        progress_status["current_clause"] = i
        progress_status["estimated_seconds_remaining"] = remaining

    df = pd.DataFrame(results)

    output_path = "results/contract_analysis_results.csv"

    df.to_csv(output_path, index=False)

    progress_status["status"] = "completed"
    progress_status["progress"] = 100
    progress_status["estimated_seconds_remaining"] = 0

# -------------------------
# STATUS ENDPOINT (NEW)
# -------------------------

@app.get("/status")
def get_status():
    return progress_status


# -------------------------
# DOWNLOAD CSV
# -------------------------

@app.get("/download")
def download_csv():

    return FileResponse(
        "results/contract_analysis_results.csv",
        media_type="text/csv",
        filename="contract_analysis_results.csv"
    )
