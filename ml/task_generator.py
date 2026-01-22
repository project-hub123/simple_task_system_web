import os
import pickle
import random
import sys
from typing import Dict

# ======================================================
# НАСТРОЙКА ПУТЕЙ
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.model_service import load_model

MODEL_PATH = os.path.join(BASE_DIR, "models", "text_ngram.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Сначала необходимо запустить ml/text_model.py")

# ======================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ======================================================

with open(MODEL_PATH, "rb") as f:
    ngram_model = pickle.load(f)

clf_data = load_model()
vectorizer = clf_data["vectorizer"]
classifier = clf_data["model"]

# ======================================================
# СООТВЕТСТВИЕ ТИПА ЗАДАНИЯ ПРЕДМЕТНОЙ ОБЛАСТИ
# ======================================================

TASK_DOMAIN = {
    "list_sum": "numbers",
    "list_even": "numbers",
    "list_sort": "numbers",
    "list_square": "numbers",
    "list_strings": "strings",
    "text_chars": "text",
    "text_words": "text"
}

# ======================================================
# ГЕНЕРАЦИЯ ТЕКСТА ФОРМУЛИРОВКИ (языковая модель)
# ======================================================

def generate_text(min_words: int = 5, max_words: int = 15) -> str:
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
# ГЕНЕРАЦИЯ ВХОДНЫХ ДАННЫХ
# ======================================================

def generate_input_data(task_type: str, domain: str) -> str:
    if domain == "numbers":
        data = [random.randint(1, 20) for _ in range(8)]
        return f"data = Список чисел: {data}"

    if domain == "strings":
        data = ["яблоко", "груша", "слива", "банан", "апельсин"]
        random.shuffle(data)
        return f"data = Список строк: {data}"

    if domain == "text":
        texts = [
            "Программирование на Python",
            "Анализ данных и машинное обучение",
            "Разработка информационных систем"
        ]
        return f'data = Строка текста: "{random.choice(texts)}"'

    return "data = Входные данные определяются пользователем."

# ======================================================
# ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ЗАДАНИЯ
# ======================================================

def generate_task() -> Dict[str, str]:
    raw_text = generate_text()

    X_vec = vectorizer.transform([raw_text])
    task_type = classifier.predict(X_vec)[0]

    domain = TASK_DOMAIN.get(task_type, "numbers")

    input_data = generate_input_data(task_type, domain)

    if domain == "numbers":
        final_text = f"Дан список чисел. {raw_text} Используйте входные данные ниже."
    elif domain == "strings":
        final_text = f"Дан список строк. {raw_text} Используйте входные данные ниже."
    else:
        final_text = f"Дана строка текста. {raw_text} Используйте входные данные ниже."

    return {
        "task_text": final_text,
        "task_type": task_type,
        "input_data": input_data
    }

# ======================================================
# ЛОКАЛЬНЫЙ ЗАПУСК ДЛЯ ПРОВЕРКИ
# ======================================================

if __name__ == "__main__":
    task = generate_task()
    print("СГЕНЕРИРОВАННОЕ ЗАДАНИЕ:")
    print(task["task_text"])
    print("ИСХОДНЫЕ ДАННЫЕ:")
    print(task["input_data"])
    print("ТИП:", task["task_type"])
