import os
import ast
import random
import joblib
import pandas as pd
from typing import Dict, Tuple

# ============================================================
# ПУТИ
# ============================================================

MODEL_PATH = "models/code_checker_model.pkl"
TASKS_PATH = "data/tasks_300.csv"
TRAIN_DATASET_PATH = "data/python_tasks_dataset.csv"

# ============================================================
# КЕШ МОДЕЛИ (КРИТИЧНО!)
# ============================================================

_model = None


def load_local_model():
    """
    Загружает ТОЛЬКО сохранённый Pipeline.
    Никаких vectorizer / fit / transform здесь быть НЕ МОЖЕТ.
    """
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("ML-модель не найдена. Сначала выполните обучение.")

        print("[ML] Загружена обученная модель")
        _model = joblib.load(MODEL_PATH)

    return _model


# ============================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ
# ============================================================

_last_task_index = None


def generate_task() -> str:
    global _last_task_index

    if not os.path.exists(TASKS_PATH):
        raise FileNotFoundError("Файл с заданиями не найден")

    # важно: delimiter=';' если у тебя CSV из Excel
    df = pd.read_csv(TASKS_PATH, sep=";")

    if "task" not in df.columns:
        raise ValueError(f"В CSV нет колонки 'task'. Найдено: {list(df.columns)}")

    df = df.dropna(subset=["task"])

    if df.empty:
        raise RuntimeError("Файл заданий пуст")

    indices = list(df.index)

    if _last_task_index in indices and len(indices) > 1:
        indices.remove(_last_task_index)

    idx = random.choice(indices)
    _last_task_index = idx

    return str(df.loc[idx, "task"])


# ============================================================
# СТАТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def static_code_analysis(code: str) -> Tuple[bool, Dict[str, bool], str]:
    features = {
        "has_loop": False,
        "has_if": False,
        "has_function": False
    }

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, features, f"Синтаксическая ошибка: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            features["has_loop"] = True
        elif isinstance(node, ast.If):
            features["has_if"] = True
        elif isinstance(node, ast.FunctionDef):
            features["has_function"] = True

    return True, features, ""


# ============================================================
# ПРОВЕРКА РЕШЕНИЯ (ГЛАВНАЯ ФУНКЦИЯ)
# ============================================================

def predict(task_text: str, solution_code: str) -> str:
    if not solution_code.strip():
        return "❌ Решение пустое."

    # 1. Синтаксис
    ok, features, error = static_code_analysis(solution_code)
    if not ok:
        return f"❌ {error}"

    # 2. ML (ТОЛЬКО PIPELINE!)
    model = load_local_model()
    ml_input = f"{task_text}\n{solution_code}"

    try:
        prediction = int(model.predict([ml_input])[0])
    except Exception as e:
        return f"❌ Ошибка проверки: {e}"

    # 3. Ответ
    if prediction == 1:
        return "✅ Решение корректное."
    else:
        return "❌ Решение некорректное."


# ============================================================
# СТАТИСТИКА МОДЕЛИ (ДЛЯ АДМИНКИ)
# ============================================================

def get_model_stats() -> Dict[str, int]:
    trained = os.path.exists(MODEL_PATH)

    if not os.path.exists(TRAIN_DATASET_PATH):
        return {"trained": trained, "records": 0, "positive": 0, "negative": 0}

    df = pd.read_csv(TRAIN_DATASET_PATH)

    if "label" not in df.columns:
        return {"trained": trained, "records": len(df), "positive": 0, "negative": 0}

    return {
        "trained": trained,
        "records": int(len(df)),
        "positive": int((df["label"] == 1).sum()),
        "negative": int((df["label"] == 0).sum())
    }
