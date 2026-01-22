import os
import pickle
import random
from typing import Dict

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.model_service import load_model

# ======================================================
# ПУТИ
# ======================================================

MODEL_PATH = os.path.join(BASE_DIR, "models", "text_ngram.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Модель генерации текста не найдена. Сначала запустите text_model.py")

# ======================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ======================================================

with open(MODEL_PATH, "rb") as f:
    ngram_model = pickle.load(f)

clf_data = load_model()
vectorizer = clf_data["vectorizer"]
classifier = clf_data["model"]

# ======================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================

def _get_valid_starters():
    starters = []
    for w1, w2 in ngram_model.keys():
        if w1 and w1[0].isalpha():
            starters.append((w1, w2))
    return starters or list(ngram_model.keys())

# ======================================================
# ГЕНЕРАЦИЯ ТЕКСТА
# ======================================================

def generate_text(min_words: int = 6, max_words: int = 20) -> str:
    starters = _get_valid_starters()
    start = random.choice(starters)

    words = [start[0], start[1]]

    while len(words) < max_words:
        key = (words[-2], words[-1])
        next_words = ngram_model.get(key)
        if not next_words:
            break
        words.append(random.choice(next_words))

    if len(words) < min_words:
        return generate_text(min_words, max_words)

    text = " ".join(words)
    text = text.strip()

    if not text.endswith("."):
        text += "."

    return text

# ======================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================

def generate_task() -> Dict[str, str]:
    """
    Генерирует новое учебное задание.
    Текст формируется языковой моделью,
    тип определяется ML-классификатором.
    """

    text = generate_text()

    X_vec = vectorizer.transform([text])
    task_type = classifier.predict(X_vec)[0]

    return {
        "task_text": text,
        "task_type": task_type,
        "input_data": ""
    }

# ======================================================
# ЛОКАЛЬНЫЙ ЗАПУСК
# ======================================================

if __name__ == "__main__":
    task = generate_task()
    print("СГЕНЕРИРОВАННОЕ ЗАДАНИЕ:")
    print(task["task_text"])
    print("ТИП:", task["task_type"])
