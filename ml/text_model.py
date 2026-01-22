import os
import pandas as pd
import random
import pickle
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "tasks_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "text_ngram.pkl")

df = pd.read_csv(DATA_PATH)

model = defaultdict(list)

for text in df["task_text"].astype(str):
    words = text.lower().split()
    for i in range(len(words) - 2):
        key = (words[i], words[i + 1])
        model[key].append(words[i + 2])

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("OK: модель генерации текста обучена и сохранена")
