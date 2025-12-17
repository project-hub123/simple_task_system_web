import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

DATA_PATH = "data/python_tasks.csv"
MODEL_PATH_V1 = "models/model_v1.pkl"
MODEL_PATH_V2 = "models/model_v2.pkl"

def train_model_v1():
    df = pd.read_csv(DATA_PATH)

    X = df["solution"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH_V1)

    return model, acc


def train_model_v2():
    df = pd.read_csv(DATA_PATH)

    X = df["solution"]
    y = df["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X, y)

    acc = model.score(X, y)

    joblib.dump(model, MODEL_PATH_V2)

    return model, acc
