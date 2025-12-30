# train_model.py
# Автор: Федотова Анастасия Алексеевна

import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model_task_classifier.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Загрузка обучающего датасета...")
    df = pd.read_csv(DATA_PATH)

    X = df["task_text"]
    y = df["task_type"]

    print("Векторизация текста...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000
    )
    X_vec = vectorizer.fit_transform(X)

    print("Обучение модели...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    print("Сохранение модели...")
    joblib.dump(
        {"vectorizer": vectorizer, "model": model},
        MODEL_PATH
    )

    print("ГОТОВО.")
    print("Модель сохранена в:", MODEL_PATH)

if __name__ == "__main__":
    main()
