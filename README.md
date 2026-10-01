# Risk Clause Classifier

This project trains and serves a clause-classification model to identify risk-related contract clauses and score overall contract risk.

## What this repository includes

- `train.py` – trains a DistilBERT classifier using `risk_clause_labelled.csv` and saves model artifacts to `risk_clause_model/`
- `predict.py` – runs single-clause prediction using the saved model
- `analyze_contract.py` – analyzes a PDF contract, classifies each clause, and writes `contract_analysis_results.csv`
- `app.py` – FastAPI service for asynchronous PDF analysis, progress tracking, and CSV download
- `extract.py` – extracts long contract clauses from SEC filings for data collection
- `risk_clause_model/` – trained model + tokenizer files
- `results/` – training checkpoints

## Dataset format

The training CSV in this repo is:

- `risk_clause_labelled.csv`

Main columns used by scripts:

- `Clause_Text`
- `Category`

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas numpy fastapi uvicorn pymupdf requests beautifulsoup4
```

## Training

Run:

```bash
python train.py
```

This saves the trained model to:

- `risk_clause_model/`

## Single clause prediction

Run:

```bash
python predict.py
```

## Full contract analysis (CLI)

Place a contract PDF at the expected path (default in script: `contract.pdf`) and run:

```bash
python analyze_contract.py
```

Output file:

- `contract_analysis_results.csv`

## API usage

Start server:

```bash
uvicorn app:app --reload
```

### Endpoints

- `POST /analyze`  
  Upload a PDF file and start background processing.

- `GET /status`  
  Check progress (`status`, percentage, current clause, remaining estimate).

- `GET /download`  
  Download generated CSV from `results/contract_analysis_results.csv`.

## Notes

- Ensure `risk_clause_model/` exists before running prediction/analysis scripts.
- Contract text is split into sentence-like clauses and then classified with confidence scores.
