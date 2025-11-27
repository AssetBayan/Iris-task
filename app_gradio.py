import requests
import gradio as gr

FASTAPI_URL = "http://127.0.0.1:8000/predict/"


def predict_species(sl: float, sw: float, pl: float, pw: float) -> str:
    """
    Клиентская функция для Gradio:
    - формирует JSON
    - делает POST на FastAPI
    - возвращает строку с результатом
    """
    payload = {"sl": sl, "sw": sw, "pl": pl, "pw": pw}

    try:
        response = requests.post(FASTAPI_URL, json=payload, timeout=5)
    except Exception as e:
        return f"Ошибка подключения к FastAPI серверу: {e}"

    if response.status_code != 200:
        return f"Ошибка от сервера: {response.status_code} - {response.text}"

    data = response.json()
    species = data.get("species", "Неизвестно")
    return f"Предсказанный вид (species): {species}"


# === Описание Gradio UI ===

with gr.Blocks(title="Iris Predictor - Gradio Client") as demo:
    gr.Markdown(
        """
    # 🌸 Iris Species Predictor (Gradio → FastAPI)

    Этот интерфейс:
    - принимает параметры цветка Iris
    - отправляет их на FastAPI backend (`/predict/`)
    - отображает предсказанный вид
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
    btn.click(predict_species, inputs=[sl, sw, pl, pw], outputs=[output])

if __name__ == "__main__":
    # share=True — для внешнего публичного линка при необходимости
    demo.launch(server_port=7860, share=False)
