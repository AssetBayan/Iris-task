import joblib
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


def train_and_save_model(model_path: str = "iris_model.pkl") -> None:
    # Загружаем встроенный датасет Iris
    iris = datasets.load_iris()
    X = iris.data  # признаки: sepal length, sepal width, petal length, petal width
    y = iris.target  # классы: 0, 1, 2

    # Простая логистическая регрессия
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Сохраняем модель
    joblib.dump(model, model_path)
    print(f"Модель сохранена в {model_path}")


if __name__ == "__main__":
    train_and_save_model()
