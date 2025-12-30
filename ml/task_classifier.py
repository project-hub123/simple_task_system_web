# ml/task_classifier.py
# Автор: Федотова Анастасия Алексеевна
# Тема ВКР:
# «Автоматическая генерация и проверка учебных заданий по языку программирования Python
#  с помощью нейронных сетей
#  (на примере ЧОУ ВО „Московский университет имени С.Ю. Витте“)»

import os
import joblib

# ======================================================
# ПУТИ
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model_task_classifier.pkl")

# ======================================================
# КЭШ
# ======================================================

_bundle = None


# ======================================================
# ЗАГРУЗКА МОДЕЛИ
# ======================================================

def load_model():
    """
    Загружает обученную ML-модель и TF-IDF векторизатор
    из одного файла model_task_classifier.pkl.
    Загрузка выполняется один раз (кэширование).
    """

    global _bundle

    if _bundle is not None:
        return _bundle

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Файл model_task_classifier.pkl не найден.\n"
            "Необходимо выполнить обучение модели командой:\n"
            "python train_model.py"
        )

    _bundle = joblib.load(MODEL_PATH)

    if "model" not in _bundle or "vectorizer" not in _bundle:
        raise RuntimeError(
            "Файл модели имеет неверную структуру. "
            "Ожидались ключи 'model' и 'vectorizer'."
        )

    return _bundle


# ======================================================
# КЛАССИФИКАЦИЯ ЗАДАНИЯ
# ======================================================

def classify_task(task_text: str) -> str:
    """
    Классифицирует текст учебного задания и возвращает его тип.

    Примеры возвращаемых значений:
        'text_count_words'
        'text_remove_spaces'
        'list_sum'
        'list_reverse'
        'strings_upper'
        'dict_sum'
        и т.д.
    """

    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("Текст задания пуст или имеет неверный формат")

    bundle = load_model()
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    X = vectorizer.transform([task_text])
    predicted_class = model.predict(X)[0]

    return predicted_class
