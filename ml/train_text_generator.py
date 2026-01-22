import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "tasks_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "text_generator.pt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

df = pd.read_csv(DATA_PATH)
text = "\n".join(df["task_text"].astype(str).str.lower())

chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

SEQ_LEN = 40
X, y = [], []

for i in range(len(text) - SEQ_LEN):
    X.append([char_to_idx[c] for c in text[i:i + SEQ_LEN]])
    y.append(char_to_idx[text[i + SEQ_LEN]])

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

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

model = TextGenerator(len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

EPOCHS = 5
BATCH = 128

for epoch in range(EPOCHS):
    for i in range(0, len(X), BATCH):
        xb = X[i:i+BATCH]
        yb = y[i:i+BATCH]

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, loss={loss.item():.4f}")

torch.save({
    "model": model.state_dict(),
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char
}, MODEL_PATH)

print("OK: PyTorch модель генерации сохранена:", MODEL_PATH)
