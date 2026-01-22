import os
import pandas as pd
import pickle
from collections import defaultdict

# ======================================================
# ПУТИ
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "tasks_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "text_ngram.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# ЗАГРУЗКА ДАТАСЕТА (устойчиво к "кривому" CSV)
# ======================================================

df = pd.read_csv(
    DATA_PATH,
    sep=",",
    quotechar='"',
    escapechar="\\",
    engine="python",
    on_bad_lines="skip"
)

if "task_text" not in df.columns:
    raise RuntimeError("В датасете отсутствует колонка task_text")

texts = df["task_text"].dropna().astype(str)

if texts.empty:
    raise RuntimeError("Колонка task_text пуста")

# ======================================================
# ОБУЧЕНИЕ ЯЗЫКОВОЙ МОДЕЛИ (n-граммы)
# ======================================================

model = defaultdict(list)

for text in texts:
    words = text.lower().split()
    if len(words) < 3:
        continue

    for i in range(len(words) - 2):
        key = (words[i], words[i + 1])
        model[key].append(words[i + 2])

if not model:
    raise RuntimeError("Не удалось обучить модель: недостаточно данных")

# ======================================================
# СОХРАНЕНИЕ МОДЕЛИ
# ======================================================

with open(MODEL_PATH, "wb") as f:
    pickle.dump(dict(model), f)

print("OK: модель генерации текста обучена и сохранена")
print("Файл модели:", MODEL_PATH)
print("Количество состояний модели:", len(model))
