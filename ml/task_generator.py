import os
import pickle
import random
from typing import Dict

from ml.model_service import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "text_ngram.pkl")

with open(MODEL_PATH, "rb") as f:
    ngram_model = pickle.load(f)

clf = load_model()
vectorizer = clf["vectorizer"]
classifier = clf["model"]

def generate_text(length=20):
    start = random.choice(list(ngram_model.keys()))
    words = [start[0], start[1]]

    for _ in range(length):
        key = tuple(words[-2:])
        next_words = ngram_model.get(key)
        if not next_words:
            break
        words.append(random.choice(next_words))

    return " ".join(words)

def generate_task() -> Dict[str, str]:
    text = generate_text()
    task_type = classifier.predict(vectorizer.transform([text]))[0]

    return {
        "task_text": text,
        "task_type": task_type,
        "input_data": ""
    }

if __name__ == "__main__":
    task = generate_task()
    print(task["task_text"])
    print("ТИП:", task["task_type"])
