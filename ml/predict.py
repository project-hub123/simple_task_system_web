import os
import ast
import joblib
from typing import Dict, Optional

# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_PATH = "models/code_checker_model.pkl"

_model = None

# ============================================================
# НОРМАЛИЗАЦИЯ КОДА (КРИТИЧЕСКИ ВАЖНО)
# ============================================================

def normalize_code(code: str) -> str:
    """
    Убирает скрытые символы, которые ломают ast.parse
    """
    return (
        code
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00A0", " ")
        .replace("\ufeff", "")
        .strip()
    )

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
# СТАТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def static_analysis(code: str) -> Dict[str, bool]:
    features = {
        "syntax_ok": True,
        "has_function": False,
        "has_return": False,
        "uses_loop": False,
        "uses_condition": False
    }

    code = normalize_code(code)

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
# ОСНОВНАЯ ПРОВЕРКА РЕШЕНИЯ
# ============================================================

def predict(solution_text: str, task_text: Optional[str] = "") -> str:
    if not solution_text or not solution_text.strip():
        return "❌ Решение пустое."

    solution_text = normalize_code(solution_text)
    task_text = normalize_code(task_text or "")

    # ---------- AST ----------
    features = static_analysis(solution_text)

    if not features["syntax_ok"]:
        return "❌ Синтаксическая ошибка в коде."

    feedback = []
    feedback.append("✅ Синтаксический анализ выполнен успешно.")

    if features["uses_loop"]:
        feedback.append("✔ Используются циклы.")
    if features["uses_condition"]:
        feedback.append("✔ Используются условия.")
    if features["has_function"]:
        feedback.append("✔ Объявлена функция.")
    if features["has_return"]:
        feedback.append("✔ Используется return.")

    # ---------- ML ----------
    try:
        model = load_model()
        ml_input = task_text + "\n" + solution_text
        prediction = int(model.predict([ml_input])[0])
    except Exception as e:
        return f"❌ Ошибка ML-модели: {e}"

    feedback.append("")
    feedback.append("📊 Результат машинного анализа:")

    if prediction == 1:
        feedback.append("✅ Решение признано корректным.")
    else:
        feedback.append("❌ Решение признано некорректным.")

    return "\n".join(feedback)
