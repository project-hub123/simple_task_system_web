import os
import ast
import joblib
import pandas as pd
from typing import Dict, Tuple

# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_PATH = "models/code_checker_model.pkl"
DATASET_PATH = "data/python_tasks_dataset.csv"

# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_local_model():
    """
    Загружает обученную локальную модель машинного обучения.
    Модель используется для оценки корректности решения
    Python-задачи на основе текста задания и программного кода.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Модель не найдена. Сначала выполните обучение (train.py)."
        )

    print("[ML] Загрузка обученной модели")
    return joblib.load(MODEL_PATH)

# ============================================================
# СТАТИЧЕСКИЙ АНАЛИЗ PYTHON-КОДА
# ============================================================

def static_code_analysis(code: str) -> Tuple[bool, Dict[str, bool], str]:
    """
    Выполняет синтаксический и структурный анализ Python-кода.

    Возвращает:
    - флаг корректности синтаксиса
    - словарь с найденными конструкциями
    - текстовое сообщение об ошибке (если есть)
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
# ОСНОВНАЯ ФУНКЦИЯ ПРОВЕРКИ РЕШЕНИЯ
# ============================================================

def predict_local_feedback(model, task_text: str, solution_code: str) -> str:
    """
    Проверяет решение задачи, используя гибридный подход:
    1) статический анализ Python-кода
    2) ML-классификацию корректности решения

    Возвращает развернутую текстовую обратную связь.
    """

    if not solution_code.strip():
        return "❌ Решение не содержит кода. Введите программный код для проверки."

    # ---------- Статический анализ ----------
    syntax_ok, features, error_msg = static_code_analysis(solution_code)

    if not syntax_ok:
        return f"❌ {error_msg}"

    feedback = []
    feedback.append("✅ Синтаксический анализ пройден успешно.")

    # ---------- Анализ структуры ----------
    if features["has_function"]:
        feedback.append("✔ В коде обнаружено определение функции.")
    else:
        feedback.append("❌ В коде не найдено определение функции.")

    if features["has_return"]:
        feedback.append("✔ Используется оператор return.")
    else:
        feedback.append("❌ Оператор return отсутствует.")

    if features["uses_loop"]:
        feedback.append("ℹ Используются циклы.")
    if features["uses_condition"]:
        feedback.append("ℹ Используются условные конструкции.")
    if features["uses_import"]:
        feedback.append("ℹ Используются импортируемые модули.")

    # ---------- ML-анализ ----------
    try:
        ml_input = task_text + " " + solution_code
        prediction = model.predict([ml_input])[0]
    except Exception as e:
        return f"❌ Ошибка при работе ML-модели: {e}"

    feedback.append("")
    feedback.append("📊 Результат машинного обучения:")

    if int(prediction) == 1:
        feedback.append("✅ Решение классифицировано как корректное.")
        feedback.append("📌 Итог: решение соответствует требованиям задания.")
    else:
        feedback.append("❌ Решение классифицировано как некорректное.")
        feedback.append("📌 Итог: решение требует доработки.")

    return "\n".join(feedback)

# ============================================================
# ОЦЕНКА КАЧЕСТВА МОДЕЛИ (ДЛЯ АДМИНКИ / ОТЧЁТА)
# ============================================================

def evaluate_model(model) -> Dict[str, float]:
    """
    Выполняет оценку качества обученной модели
    на всём доступном датасете.
    """

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Датасет для оценки не найден")

    df = pd.read_csv(DATASET_PATH)

    required = {"task_text", "solution_code", "label"}
    if not required.issubset(df.columns):
        raise ValueError("Некорректная структура датасета")

    df["input"] = df["task_text"] + " " + df["solution_code"]

    y_true = df["label"].astype(int)
    y_pred = model.predict(df["input"])

    accuracy = float((y_true == y_pred).mean())

    return {
        "accuracy": round(accuracy, 3),
        "records": int(len(df))
    }

# ============================================================
# СТАТИСТИКА ПО ДАННЫМ И МОДЕЛИ
# ============================================================

def get_model_stats() -> Dict[str, int]:
    """
    Возвращает статистику по обученной модели и датасету.
    Используется для отображения в административной панели.
    """

    if not os.path.exists(DATASET_PATH):
        return {
            "trained": os.path.exists(MODEL_PATH),
            "records": 0,
            "positive": 0,
            "negative": 0
        }

    df = pd.read_csv(DATASET_PATH)

    if "label" not in df.columns:
        return {
            "trained": os.path.exists(MODEL_PATH),
            "records": len(df),
            "positive": 0,
            "negative": 0
        }

    return {
        "trained": os.path.exists(MODEL_PATH),
        "records": int(len(df)),
        "positive": int((df["label"] == 1).sum()),
        "negative": int((df["label"] == 0).sum())
    }
