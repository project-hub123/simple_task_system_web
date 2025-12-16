import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


MODEL_PATH = "model.pkl"
DATASET_PATH = "python_task_dataset.csv"


def load_local_model():
    if os.path.exists(MODEL_PATH):
        print("ML: обученная модель найдена, загрузка model.pkl")
        return joblib.load(MODEL_PATH)

    print("ML: обученная модель не найдена")
    print("ML: начато обучение модели")

    df = pd.read_csv(DATASET_PATH)

    if not {'task', 'solution', 'label'}.issubset(df.columns):
        raise ValueError("Датасет должен содержать колонки task, solution, label")

    df['input'] = df['task'] + "\n" + df['solution']

    X_train, _, y_train, _ = train_test_split(
        df['input'],
        df['label'],
        test_size=0.2,
        random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)

    print("ML: обучение завершено")
    print("ML: модель сохранена в model.pkl")

    return pipeline


def predict_local_feedback(model, task, solution):
    text = task + "\n" + solution
    prediction = model.predict([text])[0]

    if prediction == 1:
        return "Решение корректное. Основные требования задания выполнены."
    else:
        return "Решение содержит ошибки или не соответствует условиям задания."
