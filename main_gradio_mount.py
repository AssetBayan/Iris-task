from typing import Any, Dict

import joblib
import numpy as np
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel, Field


# === Pydantic-модель входных данных ===

class IrisFeatures(BaseModel):
    sl: float = Field(..., description="Sepal length")
    sw: float = Field(..., description="Sepal width")
    pl: float = Field(..., description="Petal length")
    pw: float = Field(..., description="Petal width")


# === Загрузка модели ===

MODEL_PATH = "iris_model.pkl"
model = joblib.load(MODEL_PATH)

TARGET_NAMES = np.array(["setosa", "versicolor", "virginica"], dtype=object)

# === FastAPI-приложение ===

app = FastAPI(
    title="FastAPI + Gradio (Single Server)",
    description="Пример монтирования Gradio в FastAPI (один сервер).",
    version="1.0.0",
)


@app.post("/api/predict")
def predict_api(features: IrisFeatures) -> Dict[str, Any]:
    """
    POST /api/predict

    Используется как backend endpoint для Gradio
    (и можно вызывать напрямую).
    """
    data = np.array([[features.sl, features.sw, features.pl, features.pw]])
    pred_idx: int = int(model.predict(data)[0])
    species: str = str(TARGET_NAMES[pred_idx])

    return {
        "input": features.dict(),
        "prediction_index": pred_idx,
        "species": species,
    }


# === Gradio UI, смонтированное в тот же FastAPI ===

def predict_from_ui(sl: float, sw: float, pl: float, pw: float) -> str:
    """
    Функция для Gradio, использует ту же модель напрямую,
    но концептуально отображает POST /api/predict.
    """
    data = np.array([[sl, sw, pl, pw]])
    pred_idx: int = int(model.predict(data)[0])
    species: str = str(TARGET_NAMES[pred_idx])
    return f"Предсказанный вид (species): {species}"


with gr.Blocks(title="Iris Predictor - Mounted Gradio") as gradio_app:
    gr.Markdown(
        """
    # 🌸 Iris Predictor (FastAPI + Gradio, один сервер)

    - Backend endpoint: **POST /api/predict**
    - Gradio UI смонтировано на **/gradio**
    """
    )

    with gr.Row():
        sl = gr.Slider(4.0, 8.0, value=5.1, label="Sepal Length (sl)")
        sw = gr.Slider(2.0, 4.5, value=3.5, label="Sepal Width (sw)")
    with gr.Row():
        pl = gr.Slider(1.0, 7.0, value=1.4, label="Petal Length (pl)")
        pw = gr.Slider(0.1, 2.5, value=0.2, label="Petal Width (pw)")

    output = gr.Textbox(label="Результат предсказания")
    btn = gr.Button("Предсказать вид")
    btn.click(predict_from_ui, inputs=[sl, sw, pl, pw], outputs=[output])


# Монтируем Gradio-приложение внутрь FastAPI на путь /gradio
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")


if __name__ == "__main__":
    # Обычно запускают через:
    # uvicorn main_gradio_mount:app --reload --port 8000
    import uvicorn

    uvicorn.run("main_gradio_mount:app", host="127.0.0.1", port=8000, reload=True)
