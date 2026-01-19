import csv
import os
import random
from typing import Dict, List

# ======================================================
# ПУТИ К ФАЙЛАМ
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

    Ожидаемые колонки:
        task_text   — текст задания
        task_type   — тип задания из датасета
        input_data  — входные данные
    """

    if _tasks_cache:
        return

    if not os.path.exists(TASKS_FILE):
        raise RuntimeError(f"Файл с заданиями не найден: {TASKS_FILE}")

    with open(TASKS_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {"task_text", "task_type", "input_data"}
        if not required_columns.issubset(reader.fieldnames):
            raise RuntimeError(
                "CSV-файл должен содержать колонки: "
                "task_text, task_type, input_data"
            )

        for row in reader:
            task_text = row.get("task_text", "").strip()
            task_type = row.get("task_type", "").strip()
            input_data = row.get("input_data", "").strip()

            if task_text and task_type:
                _tasks_cache.append({
                    "task_text": task_text,
                    "task_type": task_type,
                    "input_data": input_data
                })

    if not _tasks_cache:
        raise RuntimeError("Файл с заданиями пуст или содержит некорректные данные")

# ======================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ
# ======================================================

def generate_task() -> Dict[str, str]:
    """
    Возвращает случайное задание из датасета.

    Формат:
    {
        "task_text": "...",
        "task_type": "...",
        "input_data": "..."
    }
    """

    load_tasks()
    return random.choice(_tasks_cache)

# ======================================================
# ЛОКАЛЬНЫЙ ТЕСТ
# ======================================================

if __name__ == "__main__":
    for _ in range(5):
        task = generate_task()
        print("Задание:")
        print(task["task_text"])
        print("Тип задания:", task["task_type"])
        print("Входные данные:", task["input_data"])
        print("-" * 50)
