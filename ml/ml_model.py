import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

MODEL_PATH = "models/model_v2.pkl"
DATASET_PATH = "data/bi_cleaning_dataset.csv"


def load_local_model():
    if os.path.exists(MODEL_PATH):
        print("ML: модель загружена")
        return joblib.load(MODEL_PATH)

    raise FileNotFoundError("ML: модель не найдена. Сначала запусти обучение.")


def predict_local_feedback(model, task, solution):
    text = task + "\n" + solution
    prediction = model.predict([text])[0]

    if prediction == 1:
        return "Решение корректное. Условие выполнено."
    return "Решение некорректное. Найдены ошибки."


def evaluate_model(model):
    df = pd.read_csv(DATASET_PATH, encoding="cp1251")
    df["input"] = df["task"] + "\n" + df["solution"]

    y_true = df["label"]
    y_pred = model.predict(df["input"])

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "records": len(df)
    }


def get_model_stats():
    df = pd.read_csv(DATASET_PATH)

    return {
        "records": len(df),
        "positive": int((df["label"] == 1).sum()),
        "negative": int((df["label"] == 0).sum()),
        "model_exists": os.path.exists(MODEL_PATH)
    }
