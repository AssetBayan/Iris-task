# Iris-task
Iris-task
Gradio + FastAPI — Machine Learning Mini Project

Separate Servers Implementation
Kyungbok University — Big Data Department
Autumn 2025 • AI Model Operations (MLOps)

1. Overview

This project demonstrates deployment of a machine learning model using two separate servers:

FastAPI – backend server providing the prediction API
Gradio – frontend interface for user interaction

The model predicts the species of an Iris flower using four numeric features:
sepal length, sepal width, petal length, and petal width.

2. Folder Structure
iris-fastapi-gradio/
│
├── train_model.py       # Trains the model and saves iris_model.pkl
├── api.py               # FastAPI backend server (prediction API)
├── app_gradio.py        # Gradio frontend interface (UI calling FastAPI)
├── iris_model.pkl       # Saved trained model
└── requirements.txt     # Project dependencies

3. Setup and Installation
3.1 Create Virtual Environment

Windows (PowerShell):

python -m venv .venv
. .\.venv\Scripts\Activate.ps1


macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

3.2 Install Dependencies
pip install fastapi uvicorn gradio scikit-learn joblib pydantic requests

4. Model Training

Run the script below to train the model and generate iris_model.pkl:

python train_model.py


A trained model file will be saved in the project directory.

5. Running the Project
5.1 Start FastAPI Backend
uvicorn api:app --reload --port 8000


Available endpoints:

API Documentation: http://127.0.0.1:8000/docs

Health Check (if enabled): http://127.0.0.1:8000/healthz

5.2 Start Gradio Frontend

Open a new terminal window:

python app_gradio.py


Gradio interface:

http://127.0.0.1:7860

The interface allows users to adjust feature sliders and obtain the predicted flower species.

6. API Example
POST /predict/

Request:

{
  "sl": 5.1,
  "sw": 3.5,
  "pl": 1.4,
  "pw": 0.2
}


Response:

{
  "prediction": "setosa",
  "proba": {
    "setosa": 0.981,
    "versicolor": 0.019,
    "virginica": 0.000
  }
}

7. Component Description
File	Description
train_model.py	Trains the machine learning model and saves it as .pkl
api.py	FastAPI backend providing the /predict/ endpoint
app_gradio.py	Gradio interface sending user input to FastAPI
requirements.txt	Python dependency list
8. System Status
Component	Status	URL
FastAPI Server	Ready	http://127.0.0.1:8000/docs

Gradio UI	Ready	http://127.0.0.1:7860

Model File	Ready	iris_model.pkl
9. Author

Asset Bayan
Big Data Department, Kyungbok University
Autumn 2025 — MLOps Mini Project
