import os
import ast
import joblib
import numpy as np

MODEL_PATH = "models/code_checker_model.pkl"

_model = None


# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_model():
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("ML-модель не найдена")

        _model = joblib.load(MODEL_PATH)

    return _model


# ============================================================
# СИНТАКСИЧЕСКИЙ АНАЛИЗ
# ============================================================

def static_analysis(code: str):
    if not isinstance(code, str):
        return False, "Код не является строкой"

    if not code.strip():
        return False, "Код пустой"

    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Синтаксическая ошибка: {e}"

    return True, "OK"


# ============================================================
# ОСНОВНАЯ ПРОВЕРКА
# ============================================================

def predict(solution_text: str, task_text: str = "") -> str:
    # --- 1. СИНТАКСИС ---
    ok, msg = static_analysis(solution_text)
    if not ok:
        return f"❌ {msg}"

    feedback = []
    feedback.append("✅ Синтаксический анализ выполнен успешно.")

    # --- 2. СТРУКТУРНЫЙ АНАЛИЗ ---
    tree = ast.parse(solution_text)

    has_loop = False
    has_condition = False
    has_function = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            has_loop = True
        if isinstance(node, ast.If):
            has_condition = True
        if isinstance(node, ast.FunctionDef):
            has_function = True

    feedback.append("📐 Анализ структуры:")
    feedback.append("✔ Используются циклы." if has_loop else "ℹ Циклы не используются.")
    feedback.append("✔ Используются условия." if has_condition else "ℹ Условия не используются.")
    feedback.append("✔ Объявлена функция." if has_function else "ℹ Функция не объявлена.")

    # --- 3. ML ПРОВЕРКА ---
    try:
        model = load_model()
        ml_input = f"{task_text}\n{solution_text}"
        prediction = int(model.predict([ml_input])[0])
    except Exception as e:
        feedback.append("")
        feedback.append(f"⚠ Ошибка ML-модуля: {e}")
        return "\n".join(feedback)

    feedback.append("")
    feedback.append("🧠 Результат машинного анализа:")

    if prediction == 1:
        feedback.append("✅ Решение классифицировано как корректное.")
    else:
        feedback.append("❌ Решение классифицировано как некорректное.")

    return "\n".join(feedback)
