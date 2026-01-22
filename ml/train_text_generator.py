import os
import pandas as pd
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "text_generator.h5")
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "tasks_dataset.csv")

model = tf.keras.models.load_model(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
texts = "\n".join(df["task_text"].astype(str).str.lower())

chars = sorted(list(set(texts)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

def generate_task(seed: str = "дан", length: int = 200) -> str:
    seed = seed.lower()
    result = seed

    for _ in range(length):
        seq = result[-40:]
        seq_idx = [char_to_idx.get(c, 0) for c in seq]
        seq_idx = np.expand_dims(seq_idx, axis=0)

        preds = model.predict(seq_idx, verbose=0)[0]
        next_char = idx_to_char[np.argmax(preds)]
        result += next_char

    return result.strip()


if __name__ == "__main__":
    print(generate_task())
