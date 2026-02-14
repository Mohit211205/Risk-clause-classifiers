<<<<<<< HEAD
# Risk-clause-classifiers
=======
---
license: mit
---
>>>>>>> 17efa33e6a905f74adced154117a7489f1672bab


# Risk Clause Classifier

A lightweight AI system for classifying legal contract clauses into:

- Clause Category (10 classes)
- Risk Level (Low / Medium / High)

This project is designed for low-resource environments and CPU-based execution.

## 🔍 Project Overview

The system uses a distilled domain-specific language model to generate clause embeddings and applies lightweight classifiers for:

- Category Prediction
- Risk Assessment

The goal is to build an efficient, scalable, and hackathon-ready legal clause analysis tool.

## 🏗 Architecture

Clause Text  
→ Transformer Embeddings  
→ Category Classifier  
→ Risk Level Classifier  

## ⚙ Tech Stack

- Python
- Transformers
- Hugging Face Hub
- Scikit-learn
- FastAPI
- Git & GitHub

## 📂 Repository Structure

- `/src` – Training & inference scripts  
- `/models` – Saved classifier models  
- `/app.py` – API server  
- `requirements.txt` – Dependencies  

## 🚀 Deployment

The model files are hosted on Hugging Face.  
Codebase is maintained on GitHub for team collaboration.

## 👥 Team Collaboration

- GitHub → Code management  
- Hugging Face → Model storage  
- Multi-device sync enabled via Git remotes  

## 📌 Hackathon Focus

- CPU-friendly design  
- Lightweight inference  
- Structured JSON output  
- Legal domain specialization  

---

Built for efficient legal risk clause analysis.
