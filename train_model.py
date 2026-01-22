# train_model.py
# Автор: Федотова Анастасия Алексеевна

import os
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ml.model_service import save_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

DATA_PATH = os.path.join(DATA_DIR, "train_data.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

REQUIRED_COLUMNS = {"task_text", "task_type"}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def validate_dataset(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Отсутствуют обязательные колонки: {', '.join(missing)}"
        )


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Не найден обучающий датасет: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)
    return df


def main():
    print("Загрузка обучающего датасета...")
    df = load_dataset()

    X = df["task_text"]
    y = df["task_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Векторизация текста...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Обучение нейросетевой модели MLPClassifier...")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"MLPClassifier accuracy: {accuracy:.4f}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "MLPClassifier": {
                    "accuracy": accuracy
                }
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    save_model({
        "vectorizer": vectorizer,
        "model": model,
        "model_name": "MLPClassifier",
        "accuracy": accuracy
    })

    print("ГОТОВО. Нейросетевая модель обучена и сохранена.")


if __name__ == "__main__":
    main()
