import joblib
from pathlib import Path

# ======================================================
# СЕРВИС РАБОТЫ С МОДЕЛЬЮ (ИСПОЛЬЗУЕТСЯ ТОЛЬКО ДЛЯ ОБУЧЕНИЯ / ОТЧЁТА)
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model_task_classifier.pkl"


def save_model(model_data: dict):
    """
    Сохраняет обученную модель классификатора.
    Используется ТОЛЬКО в процессе обучения.
    """
    joblib.dump(model_data, MODEL_PATH)


def load_model() -> dict | None:
    """
    Загружает модель классификатора.
    В runtime генерации заданий НЕ используется.
    """
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def model_exists() -> bool:
    """
    Проверка наличия модели (для интерфейса обучения).
    """
    return MODEL_PATH.exists()
