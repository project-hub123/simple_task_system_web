import os
import pandas as pd
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "text_generator.h5")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "tasks_dataset.csv")

model = tf.keras.models.load_model(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
texts = "\n".join(df["task_text"].astype(str).str.lower())

chars = sorted(list(set(texts)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

def generate_task(seed="дан", length=200):
    result = seed.lower()
    for _ in range(length):
        seq = result[-40:]
        seq_idx = [char_to_idx.get(c, 0) for c in seq]
        seq_idx = np.expand_dims(seq_idx, axis=0)
        preds = model.predict(seq_idx, verbose=0)[0]
        result += idx_to_char[np.argmax(preds)]
    return result.strip()
