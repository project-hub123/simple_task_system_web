# task_generator.py
# Автор: Федотова Анастасия Алексеевна
# Тема ВКР:
# «Автоматическая генерация и проверка учебных заданий по языку программирования Python
#  с помощью нейронных сетей (на примере ЧОУ ВО „Московский университет имени С.Ю. Витте“)»

import csv
import os
import random
from typing import Dict, List

from ml.task_classifier import classify_task

# ======================================================
# ПУТИ К ФАЙЛАМ ПРОЕКТА
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TASKS_FILE = os.path.join(DATA_DIR, "tasks_dataset.csv")

# ======================================================
# КЭШ ЗАДАНИЙ
# ======================================================

_tasks_cache: List[Dict[str, str]] = []

# ======================================================
# ЗАГРУЗКА ЗАДАНИЙ ИЗ CSV
# ======================================================

def load_tasks() -> None:
    """
    Загружает задания из CSV-файла в память.
    Формат CSV:
        task_id, task_text
    """

    if _tasks_cache:
        return

    if not os.path.exists(TASKS_FILE):
        raise RuntimeError(
            f"Файл с заданиями не найден: {TASKS_FILE}"
        )

    with open(TASKS_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "task_text" not in reader.fieldnames:
            raise RuntimeError(
                "CSV-файл должен содержать колонку 'task_text'"
            )

        for row in reader:
            text = row.get("task_text", "").strip()
            if text:
                _tasks_cache.append({
                    "task_text": text
                })

    if not _tasks_cache:
        raise RuntimeError("Файл с заданиями пуст")


# ======================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ
# ======================================================

def generate_task() -> Dict[str, str]:
    """
    Выбирает случайное задание и определяет его тип
    с помощью обученной ML-модели.

    Возвращает словарь:
    {
        "task_text": "...",
        "task_type": "list_sum" | "text_count" | ...
    }
    """

    load_tasks()

    task = random.choice(_tasks_cache)
    task_text = task["task_text"]

    # ML-классификация задания
    task_type = classify_task(task_text)

    return {
        "task_text": task_text,
        "task_type": task_type
    }


# ======================================================
# ЛОКАЛЬНЫЙ ТЕСТ
# ======================================================

if __name__ == "__main__":
    task = generate_task()
    print("Задание:")
    print(task["task_text"])
    print("\nТип задания:", task["task_type"])
