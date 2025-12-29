import csv
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "tasks_300_labeled.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "task_vectorizer.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "task_classifier.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    texts = []
    labels = []

    with open(DATASET_PATH, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            texts.append(row[0])
            labels.append(row[1])

    return texts, labels


def train():
    texts, labels = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifier = LogisticRegression(
        max_iter=2000,
        solver="liblinear"
    )

    classifier.fit(X_train_vec, y_train)

    y_pred = classifier.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"[TASK CLASSIFIER] accuracy = {acc:.3f}")

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)

    print("Модель классификации заданий сохранена")


if __name__ == "__main__":
    train()
