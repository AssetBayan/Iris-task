# Iris-task
Gradio + FastAPI — Machine Learning Mini Project

Separate Servers Implementation
Kyungbok University — Big Data Department
Autumn 2025 • AI Model Operations (MLOps)

Overview

This project demonstrates how to deploy a machine learning model using two separate servers:

FastAPI — backend server providing a prediction API

Gradio — frontend interface for user interaction

The model predicts the species of an Iris flower based on four numeric features:
sepal length, sepal width, petal length, and petal width.

Folder Structure
iris-fastapi-gradio/
│
├── train_model.py       # Trains the model and saves iris_model.pkl
├── api.py               # FastAPI backend server (prediction API)
├── app_gradio.py        # Gradio frontend interface (UI calling FastAPI)
├── iris_model.pkl       # Saved trained model
└── requirements.txt     # Project dependencies

Setup and Installation
1. Create a Virtual Environment

Windows (PowerShell):

python -m venv .venv
. .\.venv\Scripts\Activate.ps1


macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

2. Install Dependencies
pip install fastapi uvicorn gradio scikit-learn joblib pydantic requests

Model Training

Run the following script to train the model and save iris_model.pkl:

python train_model.py


After execution, a trained model file (iris_model.pkl) will be generated.

Running the Project
1. Start the FastAPI Server (Backend)
uvicorn api:app --reload --port 8000


Available URLs:

API Documentation: http://127.0.0.1:8000/docs

Health Check (if implemented): http://127.0.0.1:8000/healthz

2. Start the Gradio Application (Frontend)

Open a new terminal:

python app_gradio.py


The Gradio UI will be available at:

http://127.0.0.1:7860

Users can adjust feature values via sliders and submit them to receive predictions.

API Endpoint Example
POST /predict/

Example request:

{
  "sl": 5.1,
  "sw": 3.5,
  "pl": 1.4,
  "pw": 0.2
}


Example response:

{
  "prediction": "setosa",
  "proba": {
    "setosa": 0.981,
    "versicolor": 0.019,
    "virginica": 0.000
  }
}

Component Description
File	Description
train_model.py	Trains the machine learning model and saves it as .pkl
api.py	Loads the model and provides the FastAPI /predict/ API
app_gradio.py	Gradio interface that sends user input to the FastAPI server
requirements.txt	Python dependency list
System Status
Component	Status	URL
FastAPI Server	Ready	http://127.0.0.1:8000/docs

Gradio UI	Ready	http://127.0.0.1:7860

ML Model	Ready	iris_model.pkl
Author

Asset Bayan
Big Data Department, Kyungbok University
Autumn 2025 — MLOps Mini Project

"From model training to API and UI integration."
