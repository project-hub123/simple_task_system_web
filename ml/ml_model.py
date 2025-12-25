import os
import ast
import random
import joblib
import pandas as pd
from typing import Dict, Tuple

# ============================================================
# ПУТИ И ФАЙЛЫ
# ============================================================

MODEL_PATH = "models/code_checker_model.pkl"
TASKS_PATH = "data/tasks_300.csv"
TRAIN_DATASET_PATH = "data/python_tasks_dataset.csv"

# ============================================================
# ЗАГРУЗКА ML-МОДЕЛИ
# ============================================================

def load_local_model():
    """
    Загружает обученную ML-модель проверки решений.
    Используется ТОЛЬКО после обучения (train.py).
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("ML-модель не найдена. Сначала выполните обучение.")

    return joblib.load(MODEL_PATH)

# ============================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ (ИЗ CSV)
# ============================================================

def generate_task() -> str:
    """
    Возвращает случайное учебное задание из CSV-файла.
    """
    if not os.path.exists(TASKS_PATH):
        raise FileNotFoundError("Файл с заданиями не найден.")

    df = pd.read_csv(TASKS_PATH)

    if "task" not in df.columns:
        raise ValueError("В файле заданий нет колонки 'task'.")

    task = random.choice(df["task"].dropna().tolist())
    return str(task)

# ============================================================
# СТАТИЧЕСКИЙ АНАЛИЗ PYTHON-КОДА
# ============================================================

def static_code_analysis(code: str) -> Tuple[bool, Dict[str, bool], str]:
    """
    Выполняет синтаксический и структурный анализ Python-кода.
    """

    features = {
        "has_function": False,
        "has_return": False,
        "uses_import": False,
        "uses_loop": False,
        "uses_condition": False
    }

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, features, f"Синтаксическая ошибка: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            features["has_function"] = True
        elif isinstance(node, ast.Return):
            features["has_return"] = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            features["uses_import"] = True
        elif isinstance(node, (ast.For, ast.While)):
            features["uses_loop"] = True
        elif isinstance(node, ast.If):
            features["uses_condition"] = True

    return True, features, ""

# ============================================================
# ПРОВЕРКА РЕШЕНИЯ (ОСНОВНАЯ ЛОГИКА)
# ============================================================

def predict_local_feedback(
    model,
    task_text: str,
    solution_code: str
) -> str:
    """
    Проверка решения:
    1) AST-анализ
    2) ML-классификация корректности
    """

    if not solution_code.strip():
        return "❌ Решение пустое. Введите программный код."

    # ---------- СИНТАКСИС ----------
    syntax_ok, features, error_msg = static_code_analysis(solution_code)

    if not syntax_ok:
        return f"❌ {error_msg}"

    feedback = []
    feedback.append("✅ Синтаксический анализ пройден.")

    # ---------- СТРУКТУРА ----------
    feedback.append("📐 Анализ структуры решения:")

    feedback.append(
        "✔ Функция определена."
        if features["has_function"]
        else "❌ Функция не определена."
    )

    feedback.append(
        "✔ Используется return."
        if features["has_return"]
        else "❌ Отсутствует return."
    )

    if features["uses_loop"]:
        feedback.append("ℹ Используются циклы.")

    if features["uses_condition"]:
        feedback.append("ℹ Используются условия.")

    if features["uses_import"]:
        feedback.append("ℹ Используются импорты.")

    # ---------- ML-ПРОВЕРКА ----------
    feedback.append("")
    feedback.append("🧠 Результат машинного обучения:")

    try:
        ml_input = task_text + "\n" + solution_code
        prediction = int(model.predict([ml_input])[0])
    except Exception as e:
        return f"❌ Ошибка ML-модели: {e}"

    if prediction == 1:
        feedback.append("✅ Решение признано корректным.")
        feedback.append("📌 Итог: решение соответствует заданию.")
    else:
        feedback.append("❌ Решение признано некорректным.")
        feedback.append("📌 Итог: требуется доработка решения.")

    return "\n".join(feedback)

# ============================================================
# ОЦЕНКА МОДЕЛИ (ДЛЯ АДМИНКИ)
# ============================================================

def evaluate_model(model) -> Dict[str, float]:
    """
    Оценивает точность модели на обучающем датасете.
    """

    if not os.path.exists(TRAIN_DATASET_PATH):
        raise FileNotFoundError("Датасет для оценки не найден.")

    df = pd.read_csv(TRAIN_DATASET_PATH)

    required = {"task_text", "solution_code", "label"}
    if not required.issubset(df.columns):
        raise ValueError("Некорректная структура датасета.")

    df["input"] = df["task_text"] + "\n" + df["solution_code"]

    y_true = df["label"].astype(int)
    y_pred = model.predict(df["input"])

    accuracy = float((y_true == y_pred).mean())

    return {
        "accuracy": round(accuracy, 3),
        "records": int(len(df))
    }

# ============================================================
# СТАТИСТИКА
# ============================================================

def get_model_stats() -> Dict[str, int]:
    """
    Возвращает статистику по датасету и модели.
    """

    trained = os.path.exists(MODEL_PATH)

    if not os.path.exists(TRAIN_DATASET_PATH):
        return {
            "trained": trained,
            "records": 0,
            "positive": 0,
            "negative": 0
        }

    df = pd.read_csv(TRAIN_DATASET_PATH)

    if "label" not in df.columns:
        return {
            "trained": trained,
            "records": len(df),
            "positive": 0,
            "negative": 0
        }

    return {
        "trained": trained,
        "records": int(len(df)),
        "positive": int((df["label"] == 1).sum()),
        "negative": int((df["label"] == 0).sum())
    }
