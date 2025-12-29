import csv
import random
import os
from typing import Dict, List

from ml.task_classifier import classify_task

# ============================================================
# ПУТИ (СТАБИЛЬНО ДЛЯ RENDER)
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TASKS_CSV_PATH = os.path.join(
    PROJECT_ROOT, "data", "tasks_300.csv"
)

# ============================================================
# КЭШ
# ============================================================

_tasks_cache: List[str] = []

# ============================================================
# ЗАГРУЗКА CSV
# ============================================================

def _load_tasks() -> None:
    if _tasks_cache:
        return

    if not os.path.exists(TASKS_CSV_PATH):
        raise RuntimeError(
            f"Файл с заданиями не найден: {TASKS_CSV_PATH}"
        )

    with open(TASKS_CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                _tasks_cache.append(row[0].strip())

    if not _tasks_cache:
        raise RuntimeError("CSV-файл с заданиями пуст")

# ============================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ
# ============================================================

def generate_task() -> Dict[str, str]:
    """
    Генерирует учебное задание и автоматически
    определяет его тип с помощью нейросети.
    """

    _load_tasks()

    task_text = random.choice(_tasks_cache)

    try:
        task_type = classify_task(task_text)
    except Exception as e:
        raise RuntimeError(
            f"Ошибка ML-классификации задания: {e}"
        )

    return {
        "task_text": task_text,
        "task_type": task_type
    }
