"""
gradio_fastapi_twoservers.py
---------------------------
Этот файл-конспект поясняет схему "двух серверов":

1) FastAPI backend (api.py)
   - ML-модель (iris_model.pkl)
   - Endpoint: POST http://127.0.0.1:8000/predict/

2) Gradio frontend (app_gradio.py)
   - UI для пользователя
   - отправляет запросы на FastAPI backend

=== Как запускать (теоретически) ===

# 1. Запуск FastAPI backend:
uvicorn api:app --reload --port 8000

# 2. Запуск Gradio frontend:
python app_gradio.py

После этого:
- Backend доступен по: http://127.0.0.1:8000
- Swagger-документация FastAPI: http://127.0.0.1:8000/docs
- Gradio UI: http://127.0.0.1:7860

Frontend (Gradio) общается с backend (FastAPI) через HTTP POST:
    requests.post("http://127.0.0.1:8000/predict/", json=...)

Таким образом:
- BackEnd = api.py
- FrontEnd = app_gradio.py
- Связь через REST API.
"""

# Здесь можно ничего не писать, т.к. файл используется как
# пояснительная "snippet-документация" для задания.
if __name__ == "__main__":
    print("См. комментарии в файле: это демонстрация архитектуры двух серверов.")
