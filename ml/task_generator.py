import csv
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

# ======================================================
# ПУТИ
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TASKS_FILE = os.path.join(DATA_DIR, "tasks_dataset.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "text_generator.h5")

# ======================================================
# ЗАГРУЗКА ДАННЫХ ДЛЯ ВОССТАНОВЛЕНИЯ СЛОВАРЯ
# ======================================================

df = pd.read_csv(TASKS_FILE)
texts = "\n".join(df["task_text"].astype(str).str.lower())

chars = sorted(list(set(texts)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# ======================================================
# ЗАГРУЗКА МОДЕЛИ
# ======================================================

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Модель генерации текста не найдена")

model = tf.keras.models.load_model(MODEL_PATH)

# ======================================================
# ГЕНЕРАЦИЯ ТЕКСТА ЗАДАНИЯ (НЕЙРОСЕТЬ)
# ======================================================

def _generate_text(seed: str = "дан", length: int = 200) -> str:
    result = seed.lower()

    for _ in range(length):
        seq = result[-40:]
        seq_idx = [char_to_idx.get(c, 0) for c in seq]
        seq_idx = np.expand_dims(seq_idx, axis=0)

        preds = model.predict(seq_idx, verbose=0)[0]
        next_char = idx_to_char[int(np.argmax(preds))]
        result += next_char

    return result.strip()

# ======================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ (ОСНОВНАЯ ФУНКЦИЯ)
# ======================================================

def generate_task() -> Dict[str, str]:
    """
    Генерирует новое задание с помощью нейросети.
    Текст задания создаётся нейросетью, обученной на датасете.
    """

    task_text = _generate_text()

    return {
        "task_text": task_text,
        "task_type": "generated_by_neural_network",
        "input_data": ""
    }

# ======================================================
# ЛОКАЛЬНЫЙ ТЕСТ
# ======================================================

if __name__ == "__main__":
    task = generate_task()
    print("СГЕНЕРИРОВАННОЕ ЗАДАНИЕ:")
    print(task["task_text"])
