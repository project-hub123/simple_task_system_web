import os
import ast
import joblib
from typing import Optional, Dict

# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_PATH = "models/code_checker_model.pkl"

_model = None  # кеш модели

# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_model():
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("ML-модель не найдена")

        print("[ML] Загружаем pipeline модели")
        _model = joblib.load(MODEL_PATH)

    return _model

# ============================================================
# СТАТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def static_analysis(code: str) -> Dict[str, bool]:
    features = {
        "syntax_ok": True,
        "has_loop": False,
        "has_if": False
    }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        features["syntax_ok"] = False
        return features

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            features["has_loop"] = True
        if isinstance(node, ast.If):
            features["has_if"] = True

    return features

# ============================================================
# ОСНОВНАЯ ПРОВЕРКА
# ============================================================

def predict(solution_text: str, task_text: Optional[str] = "") -> str:
    if not solution_text.strip():
        return "❌ Решение пустое."

    features = static_analysis(solution_text)

    if not features["syntax_ok"]:
        return "❌ Синтаксическая ошибка в коде."

    model = load_model()

    # ВАЖНО: ровно как при обучении
    text = f"{task_text}\n{solution_text}"

    try:
        prediction = int(model.predict([text])[0])
    except Exception as e:
        return f"❌ Ошибка проверки: {e}"

    # ---------- РЕЗУЛЬТАТ ----------
    if prediction == 1:
        return (
            "✅ Решение корректное.\n"
            "Код соответствует условию задания."
        )
    else:
        return (
            "❌ Решение некорректное.\n"
            "Обнаружены логические ошибки."
        )
