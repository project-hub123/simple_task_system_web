import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model_task_classifier.pkl"


def save_model(model_data):
    joblib.dump(model_data, MODEL_PATH)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Файл модели не найден")
    return joblib.load(MODEL_PATH)


def model_exists():
    return MODEL_PATH.exists()
