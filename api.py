from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# === Модель входных данных для POST /predict/ ===


class IrisFeatures(BaseModel):
    sl: float = Field(..., description="Sepal length")
    sw: float = Field(..., description="Sepal width")
    pl: float = Field(..., description="Petal length")
    pw: float = Field(..., description="Petal width")


# === Загрузка модели ===

MODEL_PATH = "iris_model.pkl"

# В реальности модель должна быть заранее создана train_model.py
model = joblib.load(MODEL_PATH)

# Метки классов Iris (как принято в sklearn)
TARGET_NAMES = np.array(["setosa", "versicolor", "virginica"], dtype=object)

# === Создаём FastAPI-приложение ===

app = FastAPI(
    title="Iris FastAPI Server",
    description="FastAPI backend для предсказания Iris (POST /predict/).",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {
        "message": "Iris FastAPI backend. Используйте POST /predict/ для предсказаний."
    }


@app.post("/predict/")
def predict_iris(features: IrisFeatures):
    """
    POST /predict/
    Принимает JSON с полями sl, sw, pl, pw.
    Возвращает предсказанный класс.
    """
    data = np.array([[features.sl, features.sw, features.pl, features.pw]])
    pred_idx: int = int(model.predict(data)[0])
    species: str = str(TARGET_NAMES[pred_idx])

    return {
        "input": features.dict(),
        "prediction_index": pred_idx,
        "species": species,
    }
