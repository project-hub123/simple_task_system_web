import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ======================================================
# АБСОЛЮТНЫЕ ПУТИ 
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "bi_cleaning_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH_V1 = os.path.join(MODELS_DIR, "model_v1.pkl")
MODEL_PATH_V2 = os.path.join(MODELS_DIR, "model_v2.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# ======================================================
# ОБУЧЕНИЕ МОДЕЛИ V1 (train / test)
# ======================================================

def train_model_v1():
    df = pd.read_csv(DATA_PATH, sep=";")

    # имена колонок берём из bi_cleaning_dataset.csv
    X = df["solution"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            max_features=5000
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH_V1)

    print(f"ML V1: accuracy = {acc:.4f}")
    return model, acc

# ======================================================
# ОБУЧЕНИЕ МОДЕЛИ V2 
# ======================================================

def train_model_v2():
    df = pd.read_csv(DATA_PATH, sep=";")

    X = df["solution"]
    y = df["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            max_features=8000
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)

    acc = model.score(X, y)

    joblib.dump(model, MODEL_PATH_V2)

    print(f"ML V2: accuracy = {acc:.4f}")
    return model, acc

# ======================================================
# ЗАПУСК 
# ======================================================

if __name__ == "__main__":
    print("ML: запуск обучения модели V1")
    train_model_v1()

    print("ML: запуск обучения модели V2")
    train_model_v2()

    print("ML: обучение завершено")
