import os
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict

from ml.model_service import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "text_generator.pt")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "tasks_dataset.csv")

checkpoint = torch.load(MODEL_PATH)

char_to_idx = checkpoint["char_to_idx"]
idx_to_char = checkpoint["idx_to_char"]

class TextGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = TextGenerator(len(char_to_idx))
model.load_state_dict(checkpoint["model"])
model.eval()

clf = load_model()
vectorizer = clf["vectorizer"]
classifier = clf["model"]

def generate_text(seed="дан", length=200):
    result = seed.lower()

    for _ in range(length):
        seq = result[-40:]
        seq_idx = [char_to_idx.get(c, 0) for c in seq]
        seq_tensor = torch.tensor([seq_idx], dtype=torch.long)

        with torch.no_grad():
            preds = model(seq_tensor)
            next_char = idx_to_char[int(torch.argmax(preds))]

        result += next_char

    return result.strip()

def generate_task() -> Dict[str, str]:
    text = generate_text()
    X_vec = vectorizer.transform([text])
    task_type = classifier.predict(X_vec)[0]

    return {
        "task_text": text,
        "task_type": task_type,
        "input_data": ""
    }

if __name__ == "__main__":
    task = generate_task()
    print(task["task_text"])
    print("ТИП:", task["task_type"])
