# train_model.py
# Автор: Федотова Анастасия Алексеевна

import os
import requests
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

DATA_PATH = os.path.join(DATA_DIR, "train_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "model_task_classifier.pkl")

DATA_URL = "https://raw.githubusercontent.com/example-repo/datasets/main/train_data.csv"

REQUIRED_COLUMNS = {"task_text", "task_type"}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def download_dataset(url: str, save_path: str):
    print("Локальный датасет не найден. Загрузка по ссылке...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)
    print("Датасет успешно загружен.")


def validate_dataset(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Некорректная структура датасета. "
            f"Отсутствуют обязательные колонки: {', '.join(missing)}"
        )


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        download_dataset(DATA_URL, DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)
    return df


def main():
    print("Загрузка обучающего датасета...")
    df = load_dataset()

    X = df["task_text"]
    y = df["task_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Векторизация текста...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Обучение модели...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Оценка качества модели...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Сохранение модели...")
    joblib.dump(
        {"vectorizer": vectorizer, "model": model},
        MODEL_PATH
    )

    print("ГОТОВО.")
    print("Модель сохранена в:", MODEL_PATH)


if __name__ == "__main__":
    main()
