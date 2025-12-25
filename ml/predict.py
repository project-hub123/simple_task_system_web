import os
import ast
import joblib
import numpy as np
from typing import Optional, Dict

# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_PATH = "models/code_checker_model.pkl"

# Кеш модели в памяти
_model = None

# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_model():
    """
    Загружает обученную модель машинного обучения из файла.
    Используется кеширование для предотвращения повторной загрузки.
    """
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "ML-модель не найдена. Необходимо сначала выполнить обучение."
            )

        print("[ML] Загрузка модели проверки кода")
        _model = joblib.load(MODEL_PATH)

    return _model

# ============================================================
# СТАТИЧЕСКИЙ АНАЛИЗ КОДА
# ============================================================

def static_analysis(code: str) -> Dict[str, bool]:
    """
    Выполняет статический анализ Python-кода с использованием AST.

    Возвращает словарь с признаками структуры кода.
    """
    features = {
        "syntax_ok": True,
        "has_function": False,
        "has_return": False,
        "uses_loop": False,
        "uses_condition": False
    }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        features["syntax_ok"] = False
        return features

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            features["has_function"] = True
        elif isinstance(node, ast.Return):
            features["has_return"] = True
        elif isinstance(node, (ast.For, ast.While)):
            features["uses_loop"] = True
        elif isinstance(node, ast.If):
            features["uses_condition"] = True

    return features

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ ПРОГНОЗА
# ============================================================

def predict(solution_text: str, task_text: Optional[str] = "") -> str:
    """
    Проверяет корректность решения задачи по Python.

    Используется гибридный подход:
    1. Статический анализ Python-кода
    2. Классификация с помощью ML-модели

    :param solution_text: программный код решения
    :param task_text: текст задания (опционально)
    :return: развернутая текстовая обратная связь
    """

    if not solution_text or not solution_text.strip():
        return "❌ Решение пустое. Введите программный код для проверки."

    # ---------- Статический анализ ----------
    features = static_analysis(solution_text)

    if not features["syntax_ok"]:
        return "❌ Синтаксическая ошибка в коде. Проверьте корректность Python-кода."

    feedback = []
    feedback.append("✅ Синтаксический анализ выполнен успешно.")

    if features["has_function"]:
        feedback.append("✔ Обнаружено определение функции.")
    else:
        feedback.append("❌ В коде отсутствует определение функции.")

    if features["has_return"]:
        feedback.append("✔ Используется оператор return.")
    else:
        feedback.append("❌ Оператор return не найден.")

    if features["uses_loop"]:
        feedback.append("ℹ В решении используются циклы.")
    if features["uses_condition"]:
        feedback.append("ℹ В решении используются условные конструкции.")

    # ---------- ML-анализ ----------
    try:
        model = load_model()

        # ВАЖНО: формируем вход так же, как при обучении
        ml_input = f"{task_text}\n{solution_text}"

        prediction = model.predict([ml_input])[0]

        # защита от numpy типов и fallback-модели
        if isinstance(prediction, (np.integer, int)):
            prediction = int(prediction)
        else:
            prediction = int(prediction)

    except Exception as e:
        return f"❌ Ошибка при применении ML-модели: {e}"

    feedback.append("")
    feedback.append("📊 Результат машинного анализа:")

    if prediction == 1:
        feedback.append("✅ Решение классифицировано как корректное.")
        feedback.append("📌 Итог: код соответствует требованиям задания.")
    else:
        feedback.append("❌ Решение классифицировано как некорректное.")
        feedback.append("📌 Итог: обнаружены несоответствия или логические ошибки.")

    return "\n".join(feedback)
