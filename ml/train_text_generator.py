import os
import pandas as pd
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "data", "tasks_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "text_generator.h5")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

df = pd.read_csv(DATA_PATH)
texts = "\n".join(df["task_text"].astype(str).str.lower())

chars = sorted(list(set(texts)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

SEQ_LEN = 40
X, y = [], []

for i in range(len(texts) - SEQ_LEN):
    X.append([char_to_idx[c] for c in texts[i:i + SEQ_LEN]])
    y.append(char_to_idx[texts[i + SEQ_LEN]])

X = np.array(X)
y = tf.keras.utils.to_categorical(y, num_classes=len(chars))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(chars), 64),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit(X, y, epochs=5, batch_size=128)

model.save(MODEL_PATH)
print("OK: модель генерации текста сохранена:", MODEL_PATH)
