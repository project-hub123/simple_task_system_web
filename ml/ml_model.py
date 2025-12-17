import os
import ast
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

MODEL_PATH = "models/model_v2.pkl"
DATASET_PATH = "data/bi_cleaning_dataset.csv"


def load_local_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Модель для сайта не найдена. Сначала запусти обучение."
        )

    print("ML: загрузка модели для сайта")
    return joblib.load(MODEL_PATH)


# ================== ГЛАВНОЕ ИСПРАВЛЕНИЕ ==================

def predict_local_feedback(model, task, solution):
    """
    Локальная проверка решения.
    НЕ использует ML-модель, чтобы избежать падений.
    Выполняет синтаксический и структурный анализ кода.
    """

    if not solution.strip():
        return "❌ Ошибка: решение пустое."

    try:
        tree = ast.parse(solution)
    except SyntaxError as e:
        return f"❌ Синтаксическая ошибка в коде: {e}"

    used_imports = set()
    used_calls = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                used_imports.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                used_imports.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                used_calls.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                used_calls.add(node.func.id)

    feedback = []
    feedback.append("✅ Код успешно проанализирован.")
    feedback.append("")

    # Проверки под твои задания
    if "pandas" in used_imports or "pd" in solution:
        feedback.append("✔ Используется библиотека pandas.")
    else:
        feedback.append("❌ Не обнаружено использование pandas.")

    if "matplotlib" in used_imports or "plt" in solution:
        feedback.append("✔ Используется библиотека matplotlib.")
    else:
        feedback.append("❌ Не обнаружено использование matplotlib.")

    if "read_csv" in used_calls:
        feedback.append("✔ Реализовано чтение данных из CSV.")
    else:
        feedback.append("❌ Не найдено чтение CSV-файла.")

    if "groupby" in used_calls:
        feedback.append("✔ Используется группировка данных.")
    else:
        feedback.append("❌ Группировка данных не обнаружена.")

    if "plot" in used_calls:
        feedback.append("✔ Присутствует построение графиков.")
    else:
        feedback.append("❌ Построение графиков не найдено.")

    feedback.append("")
    feedback.append("📌 Итог: решение частично соответствует требованиям задания.")

    return "\n".join(feedback)


# ================== ОСТАЛЬНОЕ БЕЗ ИЗМЕНЕНИЙ ==================

def evaluate_model(model):
    df = pd.read_csv(DATASET_PATH, encoding="utf-8", encoding_errors="ignore")

    if "input" not in df.columns or "label" not in df.columns:
        raise ValueError("В датасете нет колонок input / label")

    y_true = df["label"]
    y_pred = model.predict(df["input"])

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "records": len(df)
    }


def get_model_stats():
    df = pd.read_csv(DATASET_PATH, encoding="utf-8", encoding_errors="ignore")

    if "label" not in df.columns:
        return {
            "trained": os.path.exists(MODEL_PATH),
            "records": len(df),
            "positive": 0,
            "negative": 0
        }

    return {
        "trained": os.path.exists(MODEL_PATH),
        "records": len(df),
        "positive": int((df["label"] == 1).sum()),
        "negative": int((df["label"] == 0).sum())
    }
