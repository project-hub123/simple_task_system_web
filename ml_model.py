import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

DATASET_PATH = "python_task_dataset.csv"
MODEL_PATH = "model.pkl"

_model_cache = None


def load_local_model():
    """
    Загружает модель.
    Если модели нет — обучает её на Render.
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    if os.path.exists(MODEL_PATH):
        try:
            _model_cache = joblib.load(MODEL_PATH)
            return _model_cache
        except Exception:
            pass

    # === ОБУЧЕНИЕ (если модели нет) ===
    if not os.path.exists(DATASET_PATH):
        print("Датасет не найден, модель не обучена")
        return None

    df = pd.read_csv(DATASET_PATH)

    if not {'task', 'solution', 'label'}.issubset(df.columns):
        print("Неверная структура датасета")
        return None

    df['input'] = df['task'] + "\n" + df['solution']

    X_train, _, y_train, _ = train_test_split(
        df['input'], df['label'], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)
    _model_cache = pipeline

    print("Модель обучена и сохранена")

    return _model_cache


def predict_local_feedback(model, task, solution):
    """
    Используется в app.py без изменений
    """
    if model is None:
        return "Локальная модель недоступна"

    text = task + "\n" + solution

    try:
        prediction = model.predict([text])[0]
    except Exception as e:
        return f"Ошибка предсказания: {e}"

    if prediction == 1:
        return "Решение корректное. Код соответствует заданию."
    else:
        return "В решении есть ошибки или несоответствия заданию."
