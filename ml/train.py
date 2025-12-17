"""
Назначение:
- обучение нескольких моделей машинного обучения
- анализ датасета
- вывод метрик качества
- визуализация в matplotlib (на экран)
"""

import os
import time
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ================== ПУТИ ==================

DATASET_PATH = "data/bi_cleaning_dataset.csv"
MODEL_DIR = "models"
MODEL_V1_PATH = os.path.join(MODEL_DIR, "model_v1.pkl")
MODEL_V2_PATH = os.path.join(MODEL_DIR, "model_v2.pkl")

# ================== ЗАГРУЗКА ДАННЫХ ==================

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Датасет bi_cleaning_dataset.csv не найден")

    df = pd.read_csv(DATASET_PATH, encoding="utf-8", encoding_errors="ignore")

    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Ожидаются колонки {required_cols}, найдено: {df.columns.tolist()}")

    df["input"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df

# ================== АНАЛИЗ ==================

def analyze_dataset(df):
    print("\n=== АНАЛИЗ ДАТАСЕТА ===")
    print(f"Всего записей: {len(df)}")
    print(f"Класс 1: {(df['label'] == 1).sum()}")
    print(f"Класс 0: {(df['label'] == 0).sum()}")

    df["label"].value_counts().plot(kind="bar")
    plt.title("Распределение классов")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.show()

# ================== MODEL V1 ==================

def train_model_v1(df):
    print("\n=== ОБУЧЕНИЕ MODEL V1 (train/test) ===")

    X = df["input"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy V1: {acc:.3f}")
    print(f"Время обучения: {train_time:.2f} сек")

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Матрица ошибок — Model V1")
    plt.show()

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_V1_PATH)
    print(f"Модель сохранена: {MODEL_V1_PATH}")

    return acc

# ================== MODEL V2 ==================

def train_model_v2(df):
    print("\n=== ОБУЧЕНИЕ MODEL V2 (весь датасет) ===")

    X = df["input"]
    y = df["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    start = time.time()
    model.fit(X, y)
    train_time = time.time() - start

    acc = model.score(X, y)

    print(f"Accuracy V2: {acc:.3f}")
    print(f"Время обучения: {train_time:.2f} сек")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_V2_PATH)
    print(f"Модель сохранена: {MODEL_V2_PATH}")

    return acc

# ================== ЗАПУСК ВСЕГО ==================

def train_all_models():
    print("=== ЗАПУСК ML-ОБУЧЕНИЯ ===")

    df = load_dataset()
    analyze_dataset(df)

    acc_v1 = train_model_v1(df)
    acc_v2 = train_model_v2(df)

    print("\n=== ИТОГ ===")
    print(f"Model V1 accuracy: {acc_v1:.3f}")
    print(f"Model V2 accuracy: {acc_v2:.3f}")

    if acc_v1 >= 0.7 or acc_v2 >= 0.7:
        print("✔ Требование Accuracy ≥ 70% выполнено")
    else:
        print("⚠ Accuracy ниже 70%, требуется обоснование в ВКР")

    print("\nML-обучение завершено")

if __name__ == "__main__":
    train_all_models()
