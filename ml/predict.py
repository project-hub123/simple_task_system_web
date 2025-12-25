import os
import ast
import joblib
from typing import Optional, Dict

# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_PATH = "models/code_checker_model.pkl"

_model = None

# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_model():
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("ML-модель не найдена")

        _model = joblib.load(MODEL_PATH)

    return _model

# ============================================================
# СТАТИЧЕСКИЙ АНАЛИЗ (ТОЛЬКО СИНТАКСИС)
# ============================================================

def static_analysis(code: str) -> Dict[str, bool]:
    features = {
        "syntax_ok": True,
        "uses_loop": False,
        "uses_condition": False
    }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        features["syntax_ok"] = False
        return features

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            features["uses_loop"] = True
        elif isinstance(node, ast.If):
            features["uses_condition"] = True

    return features

# ============================================================
# ОСНОВНАЯ ПРОВЕРКА (ML РЕШАЕТ ВСЁ)
# ============================================================

def predict(solution_text: str, task_text: Optional[str] = "") -> str:
    if not solution_text or not solution_text.strip():
        return "❌ Решение пустое."

    # ---------- СИНТАКСИС ----------
    features = static_analysis(solution_text)

    if not features["syntax_ok"]:
        return "❌ Синтаксическая ошибка в коде."

    feedback = []
    feedback.append("✅ Синтаксический анализ выполнен успешно.")

    if features["uses_loop"]:
        feedback.append("✔ Используются циклы.")
    if features["uses_condition"]:
        feedback.append("✔ Используются условные конструкции.")

    # ---------- ML (ОБЯЗАТЕЛЬНО) ----------
    model = load_model()   # если модель не загрузится — это ошибка разработки
    ml_input = f"{task_text}\n{solution_text}"
    prediction = int(model.predict([ml_input])[0])

    feedback.append("")
    feedback.append("📊 Результат проверки:")

    if prediction == 1:
        feedback.append("✅ Решение верное.")
    else:
        feedback.append("❌ Решение неверное.")

    return "\n".join(feedback)
