import os
import time
import json
import joblib
import pandas as pd

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
    classification_report
)

# ============================================================
# ПУТИ
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
# ЛОГ
# ============================================================

def log(msg: str):
    print(f"[TRAIN] {msg}")

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_dataset() -> pd.DataFrame:
    log("Загрузка датасета")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Датасет не найден")

    # попытка с ;
    df = pd.read_csv(
        DATASET_PATH,
        sep=";",
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip"
    )

    # если всё в одной колонке — пробуем ,
    if len(df.columns) == 1 and "," in df.columns[0]:
        log("Обнаружен CSV с разделителем ','")
        df = pd.read_csv(
            DATASET_PATH,
            sep=",",
            engine="python",
            encoding="utf-8",
            on_bad_lines="skip"
        )

    log(f"Колонки в датасете: {list(df.columns)}")

    # нормализация имён
    rename_map = {
        "task": "task_text",
        "solution": "solution_code",
        "code": "solution_code"
    }
    df.rename(columns=rename_map, inplace=True)

    required = {"task_text", "solution_code", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Некорректная структура датасета: {list(df.columns)}"
        )

    df = df.dropna(subset=list(required))

    # ===============================
    # НОРМАЛИЗАЦИЯ МЕТОК
    # ===============================
    label_map = {
        "корректно": 1,
        "правильно": 1,
        "верно": 1,
        "1": 1,
        1: 1,

        "некорректно": 0,
        "неверно": 0,
        "ошибка": 0,
        "0": 0,
        0: 0
    }

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(label_map)
    )

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["input"] = (
        df["task_text"].astype(str)
        + "\n"
        + df["solution_code"].astype(str)
    )

    log(f"Всего записей: {len(df)}")
    log(f"Корректных: {(df['label'] == 1).sum()}")
    log(f"Некорректных: {(df['label'] == 0).sum()}")

    return df

# ============================================================
# МОДЕЛЬ
# ============================================================

def build_model() -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=(1, 2)
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

# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train_model(df: pd.DataFrame):
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

    log("Обучение модели")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 3),
        "train_time_sec": round(train_time, 2),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test))
    }

    log("Метрики:")
    for k, v in metrics.items():
        log(f"{k}: {v}")

    log("Classification report:")
    print(classification_report(y_test, y_pred))

    log("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, metrics

# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def save_results(model: Pipeline, metrics: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    log(f"Модель сохранена: {MODEL_PATH}")
    log(f"Метрики сохранены: {METRICS_PATH}")

# ============================================================
# MAIN
# ============================================================

def main():
    log("=== ОБУЧЕНИЕ МОДЕЛИ ПРОВЕРКИ КОДА ===")

    df = load_dataset()
    model, metrics = train_model(df)
    save_results(model, metrics)

    log("=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")

if __name__ == "__main__":
    main()
