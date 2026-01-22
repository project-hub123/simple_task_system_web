import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from ml.model_service import load_model

# ======================================================
# ПУТИ
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TASKS_FILE = os.path.join(DATA_DIR, "tasks_dataset.csv")

TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text_generator.h5")

# ======================================================
# ЗАГРУЗКА ДАННЫХ ДЛЯ СЛОВАРЯ
# ======================================================

df = pd.read_csv(TASKS_FILE)
texts = "\n".join(df["task_text"].astype(str).str.lower())

chars = sorted(list(set(texts)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# ======================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ======================================================

if not os.path.exists(TEXT_MODEL_PATH):
    raise RuntimeError("Модель генерации текста не найдена")

text_model = tf.keras.models.load_model(TEXT_MODEL_PATH)

clf_data = load_model()
vectorizer = clf_data["vectorizer"]
classifier = clf_data["model"]

# ======================================================
# ГЕНЕРАЦИЯ ТЕКСТА
# ======================================================

def _generate_text(seed: str = "дан", length: int = 200) -> str:
    result = seed.lower()

    for _ in range(length):
        seq = result[-40:]
        seq_idx = [char_to_idx.get(c, 0) for c in seq]
        seq_idx = np.expand_dims(seq_idx, axis=0)

        preds = text_model.predict(seq_idx, verbose=0)[0]
        next_char = idx_to_char[int(np.argmax(preds))]
        result += next_char

    return result.strip()

# ======================================================
# ОСНОВНАЯ ГЕНЕРАЦИЯ ЗАДАНИЯ
# ======================================================

def generate_task() -> Dict[str, str]:
    """
    Генерирует новое учебное задание.
    Текст создаётся LSTM, тип определяется MLPClassifier.
    """

    task_text = _generate_text()

    X_vec = vectorizer.transform([task_text])
    task_type = classifier.predict(X_vec)[0]

    return {
        "task_text": task_text,
        "task_type": task_type,
        "input_data": ""
    }

# ======================================================
# ЛОКАЛЬНЫЙ ТЕСТ
# ======================================================

if __name__ == "__main__":
    task = generate_task()
    print("СГЕНЕРИРОВАННОЕ ЗАДАНИЕ:")
    print(task["task_text"])
    print("ТИП ЗАДАНИЯ (нейросеть):", task["task_type"])
