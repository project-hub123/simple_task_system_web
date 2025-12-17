import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

MODEL_PATH = "models/model_v2.pkl"
DATASET_PATH = "data/bi_cleaning_dataset.csv"

def load_local_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель для сайта не найдена. Сначала запусти обучение.")

    print("ML: загрузка модели для сайта")
    return joblib.load(MODEL_PATH)

def predict_local_feedback(model, task, solution):
    text = solution.strip()
    pred = model.predict([text])[0]

    if pred == 1:
        return "Решение корректное. Класс = 1."
    return "Решение некорректное. Класс = 0."

def evaluate_model(model):
    df = pd.read_csv(DATASET_PATH, encoding="utf-8", encoding_errors="ignore")
    df["input"] = df["text"].astype(str)

    y_true = df["label"]
    y_pred = model.predict(df["input"])

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "records": len(df)
    }

def get_model_stats():
    df = pd.read_csv(DATASET_PATH, encoding="utf-8", errors="ignore")

    return {
        "trained": os.path.exists(MODEL_PATH),
        "records": len(df),
        "positive": int((df["label"] == 1).sum()),
        "negative": int((df["label"] == 0).sum())
    }
