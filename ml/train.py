import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ============================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================

DATASET_PATH = "data/python_tasks_dataset.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

MODEL_PATH = os.path.join(MODEL_DIR, "code_checker_model.pkl")
METRICS_PATH = os.path.join(REPORT_DIR, "training_metrics.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def log(msg: str):
    """Единый формат логирования"""
    print(f"[TRAIN] {msg}")

# ============================================================
# ЗАГРУЗКА И ПРОВЕРКА ДАТАСЕТА
# ============================================================

def load_dataset() -> pd.DataFrame:
    log("Проверка наличия датасета")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Файл датасета не найден")

    log("Загрузка CSV файла")
    df = pd.read_csv(DATASET_PATH)

    log(f"Количество записей: {len(df)}")

    required_columns = {"task_text", "solution_code", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Некорректная структура датасета. "
            f"Ожидаются колонки: {required_columns}"
        )

    log("Удаление строк с пропущенными значениями")
    df = df.dropna(subset=["task_text", "solution_code", "label"])

    log("Приведение целевой переменной к целочисленному типу")
    df["label"] = df["label"].astype(int)

    log("Формирование текстового входа модели")
    df["input"] = df["task_text"] + " " + df["solution_code"]

    return df

# ============================================================
# АНАЛИЗ ДАТАСЕТА
# ============================================================

def analyze_dataset(df: pd.DataFrame):
    log("Анализ распределения классов")

    class_counts = df["label"].value_counts()
    log(f"Распределение классов:\n{class_counts}")

    plt.figure(figsize=(5, 4))
    class_counts.plot(kind="bar")
    plt.title("Распределение классов (корректно / некорректно)")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.show()

# ============================================================
# ОБУЧЕНИЕ МОДЕЛИ
# ============================================================

def build_model() -> Pipeline:
    log("Формирование ML-пайплайна")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=(1, 2),
                stop_words=None
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE
            )
        )
    ])

    return model

def train_model(df: pd.DataFrame):
    log("Разделение данных на обучающую и тестовую выборки")

    X = df["input"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = build_model()

    log("Запуск обучения модели")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    log(f"Время обучения: {train_time:.2f} сек")

    log("Получение предсказаний на тестовой выборке")
    y_pred = model.predict(X_test)

    log("Расчёт метрик качества")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "train_time_sec": round(train_time, 2),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test))
    }

    log("Метрики модели:")
    for k, v in metrics.items():
        log(f"{k}: {v}")

    log("Вывод classification report")
    print(classification_report(y_test, y_pred))

    log("Построение матрицы ошибок")
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Матрица ошибок модели проверки кода")
    plt.tight_layout()
    plt.show()

    return model, metrics

# ============================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

def save_results(model: Pipeline, metrics: dict):
    log("Создание каталогов для сохранения")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    log("Сохранение модели")
    joblib.dump(model, MODEL_PATH)

    log("Сохранение метрик обучения")
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    log(f"Модель сохранена: {MODEL_PATH}")
    log(f"Метрики сохранены: {METRICS_PATH}")

# ============================================================
# ГЛАВНАЯ ТОЧКА ВХОДА
# ============================================================

def main():
    log("=== ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ ПРОВЕРКИ PYTHON-КОДА ===")

    df = load_dataset()
    analyze_dataset(df)

    model, metrics = train_model(df)
    save_results(model, metrics)

    log("=== ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО ===")

if __name__ == "__main__":
    main()
