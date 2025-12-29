import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "task_vectorizer.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "task_classifier.pkl")

_vectorizer = None
_classifier = None


def load_model():
    global _vectorizer, _classifier

    if _vectorizer is None or _classifier is None:
        if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(CLASSIFIER_PATH):
            raise RuntimeError("Модель классификации заданий не обучена")

        _vectorizer = joblib.load(VECTORIZER_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)

    return _vectorizer, _classifier


def classify_task(task_text: str) -> str:
    vectorizer, classifier = load_model()
    X = vectorizer.transform([task_text])
    return classifier.predict(X)[0]
