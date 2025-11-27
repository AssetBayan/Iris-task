# Iris-task
# Iris Species Prediction (FastAPI + Gradio)

This project predicts Iris flower species using a trained machine learning model.  
The backend uses FastAPI and the frontend uses a Gradio UI.

---

## Author

**Asset Bayan**  
Big Data Department  
Kyungbok University  
Autumn 2025 — MLOps Mini Project

---

## Tech Stack

- Python 3.x  
- FastAPI backend  
- Gradio frontend  
- scikit-learn (Logistic Regression model)  
- pickle model storage (`.pkl`)

---

## Project Structure

/project  
├─ api.py (FastAPI Backend)  
├─ app_gradio.py (Gradio Frontend)  
├─ train_model.py (train + save iris_model.pkl)  
├─ iris_model.pkl (trained model)  
└─ README.md  

---

## Setup

```bash
pip install -r requirements.txt
Train Model (optional)
bash

python train_model.py
Run Backend (FastAPI)
bash

uvicorn api:app --reload
Open in browser:
http://127.0.0.1:8000/docs

Run Frontend (Gradio)
bash

python app_gradio.py
Open in browser:
http://127.0.0.1:7860

API Endpoint
POST /predict

Request Example
json

{"sl": 5.1, "sw": 3.5, "pl": 1.4, "pw": 0.2}
Response Example
json

{"prediction": 0, "proba": [0.98, 0.01, 0.01]}
diff

