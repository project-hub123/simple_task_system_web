import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# --- Шаг 1: Загрузка датасета ---
df = pd.read_csv("python_task_dataset.csv")

# Проверка на наличие нужных колонок
if 'task' not in df.columns or 'solution' not in df.columns or 'label' not in df.columns:
    raise ValueError("В датасете должны быть столбцы: task, solution, label")

# Объединяем task и solution в один текстовый вход
df['input'] = df['task'] + "\n" + df['solution']

# --- Шаг 2: Разделение на тренировочную и тестовую выборки ---
X_train, X_test, y_train, y_test = train_test_split(
    df['input'], df['label'], test_size=0.2, random_state=42
)

# --- Шаг 3: Создание пайплайна: TF-IDF + Логистическая регрессия ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# --- Шаг 4: Обучение модели ---
pipeline.fit(X_train, y_train)

# --- Шаг 5: Оценка качества ---
y_pred = pipeline.predict(X_test)
print("Классификационный отчёт:\n")
print(classification_report(y_test, y_pred))

# --- Шаг 6: Сохранение модели ---
joblib.dump(pipeline, 'model.pkl')
print("\nМодель успешно сохранена в model.pkl")
