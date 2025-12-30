# task_classifier.py
# Автор: Федотова Анастасия Алексеевна
# Тема ВКР:
# «Автоматическая генерация и проверка учебных заданий по языку программирования Python
#  с помощью нейронных сетей (на примере ЧОУ ВО „Московский университет имени С.Ю. Витте“)»

import os
import joblib

# ======================================================
# ПУТИ К МОДЕЛЯМ
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model_task_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "task_vectorizer.pkl")

# ======================================================
# КЭШ ЗАГРУЗКИ
# ======================================================

_model = None
_vectorizer = None


# ======================================================
# ЗАГРУЗКА МОДЕЛИ
# ======================================================

def load_model():
    """
    Загружает обученную ML-модель и TF-IDF векторизатор.
    Загрузка выполняется один раз (кэширование).
    """

    global _model, _vectorizer

    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Файл model_task_classifier.pkl не найден. "
            "Сначала обучите модель в train_model.ipynb."
        )

    if not os.path.exists(VECTORIZER_PATH):
        raise RuntimeError(
            "Файл task_vectorizer.pkl не найден. "
            "Сначала обучите модель в train_model.ipynb."
        )

    _model = joblib.load(MODEL_PATH)
    _vectorizer = joblib.load(VECTORIZER_PATH)

    return _model, _vectorizer


# ======================================================
# КЛАССИФИКАЦИЯ ЗАДАНИЯ
# ======================================================

def classify_task(task_text: str) -> str:
    """
    Классифицирует текст учебного задания и возвращает его тип.

    Пример результата:
        'text_count'
        'text_replace'
        'list_sum'
        'list_filter'
        'dict_sum'
        и т.д.
    """

    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("Текст задания пуст или имеет неверный формат")

    model, vectorizer = load_model()

    # Векторизация текста
    X = vectorizer.transform([task_text])

    # Предсказание класса
    predicted_class = model.predict(X)[0]

    return predicted_class
