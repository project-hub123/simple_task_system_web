import os

# ============================================================
# ПУТИ
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

VECTORIZER_PATH = os.path.join(
    MODEL_DIR, "task_vectorizer.pkl"
)

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR, "task_classifier.pkl"
)

# ============================================================
# ПРОВЕРКА НАЛИЧИЯ ML
# ============================================================

ML_AVAILABLE = (
    os.path.exists(VECTORIZER_PATH)
    and os.path.exists(CLASSIFIER_PATH)
)

# ============================================================
# КЛАССИФИКАЦИЯ ЗАДАНИЯ
# ============================================================

def classify_task(task_text: str) -> str:
    """
    Если ML-модель есть — используем её.
    Если нет — возвращаем базовый тип.
    """

    if not ML_AVAILABLE:
        return "general"

    # ⬇️ этот код выполнится ТОЛЬКО если модели реально есть
    import joblib

    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)

    X = vectorizer.transform([task_text])
    return classifier.predict(X)[0]
