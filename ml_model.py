import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


MODEL_PATH = "model.pkl"
DATASET_PATH = "python_task_dataset.csv"


def load_local_model():
    if os.path.exists(MODEL_PATH):
        print("ML: модель найдена, загрузка из model.pkl")
        return joblib.load(MODEL_PATH)

    print("ML: модель не найдена, запуск обучения")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("ML: датасет не найден")

    df = pd.read_csv(DATASET_PATH)

    if not {"task", "solution", "label"}.issubset(df.columns):
        raise ValueError("ML: неверная структура датасета")

    df["input"] = df["task"] + "\n" + df["solution"]

    X_train, _, y_train, _ = train_test_split(
        df["input"],
        df["label"],
        test_size=0.2,
        random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)

    print("ML: обучение завершено")
    print("ML: модель сохранена")

    return pipeline


def retrain_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("ML: старая модель удалена")
    return load_local_model()


def predict_local_feedback(model, task, solution):
    text = task + "\n" + solution
    prediction = model.predict([text])[0]

    if prediction == 1:
        return "Решение корректное. Условие задания выполнено."
    else:
        return "Решение некорректное. Обнаружены ошибки."


def evaluate_model(model):
    df = pd.read_csv(DATASET_PATH)
    df["input"] = df["task"] + "\n" + df["solution"]

    y_true = df["label"]
    y_pred = model.predict(df["input"])

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "records": len(df)
    }


def get_model_stats():
    if not os.path.exists(DATASET_PATH):
        return {"trained": False, "records": 0}

    df = pd.read_csv(DATASET_PATH)

    return {
        "trained": os.path.exists(MODEL_PATH),
        "records": len(df),
        "positive": int((df["label"] == 1).sum()),
        "negative": int((df["label"] == 0).sum())
    }
