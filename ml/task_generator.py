import os
import pickle
import random
import sys
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.model_service import load_model

# ======================================================
# ПУТИ
# ======================================================

MODEL_PATH = os.path.join(BASE_DIR, "models", "text_ngram.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Сначала запустите ml/text_model.py")

# ======================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ======================================================

with open(MODEL_PATH, "rb") as f:
    ngram_model = pickle.load(f)

clf_data = load_model()
vectorizer = clf_data["vectorizer"]
classifier = clf_data["model"]

# ======================================================
# ГЕНЕРАЦИЯ ТЕКСТА
# ======================================================

def generate_text(min_words=5, max_words=15) -> str:
    start = random.choice(list(ngram_model.keys()))
    words = [start[0], start[1]]

    while len(words) < max_words:
        key = (words[-2], words[-1])
        next_words = ngram_model.get(key)
        if not next_words:
            break
        words.append(random.choice(next_words))

    if len(words) < min_words:
        return generate_text(min_words, max_words)

    text = " ".join(words).strip()
    if not text.endswith("."):
        text += "."
    return text

# ======================================================
# ГЕНЕРАЦИЯ ВХОДНЫХ ДАННЫХ ПО ТИПУ
# ======================================================

def generate_input_data(task_type: str) -> str:
    if task_type == "list_sum":
        data = [random.randint(1, 20) for _ in range(8)]
        return f"Список чисел: {data}"

    if task_type == "list_even":
        data = [random.randint(1, 30) for _ in range(10)]
        return f"Список чисел: {data}"

    if task_type == "list_sort":
        data = [random.randint(1, 50) for _ in range(7)]
        return f"Список чисел: {data}"

    if task_type == "list_strings":
        data = ["яблоко", "груша", "слива", "банан"]
        random.shuffle(data)
        return f"Список строк: {data}"

    if task_type == "list_square":
        data = [random.randint(1, 15) for _ in range(6)]
        return f"Список чисел: {data}"

    return ""

# ======================================================
# ОСНОВНАЯ ГЕНЕРАЦИЯ ЗАДАНИЯ
# ======================================================

def generate_task() -> Dict[str, str]:
    raw_text = generate_text()

    X_vec = vectorizer.transform([raw_text])
    task_type = classifier.predict(X_vec)[0]

    input_data = generate_input_data(task_type)

    final_text = f"{raw_text} Используйте входные данные ниже."

    return {
        "task_text": final_text,
        "task_type": task_type,
        "input_data": input_data
    }

# ======================================================
# ЛОКАЛЬНЫЙ ТЕСТ
# ======================================================

if __name__ == "__main__":
    task = generate_task()
    print("СГЕНЕРИРОВАННОЕ ЗАДАНИЕ:")
    print(task["task_text"])
    print("ВХОДНЫЕ ДАННЫЕ:", task["input_data"])
    print("ТИП:", task["task_type"])
