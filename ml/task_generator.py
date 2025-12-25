import csv
import random
import os
from typing import List, Dict, Optional

# ============================================================
# НАСТРОЙКИ
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_CSV_PATH = os.path.join(BASE_DIR, "..", "data", "tasks_300.csv")
DELIMITER = ";"

# ============================================================
# ВНУТРЕННЕЕ ХРАНИЛИЩЕ
# ============================================================

_tasks_cache: List[Dict[str, str]] = []
_last_task_id: Optional[str] = None

# ============================================================
# ЗАГРУЗКА ЗАДАНИЙ ИЗ CSV
# ============================================================

def _load_tasks_from_csv() -> None:
    global _tasks_cache

    if _tasks_cache:
        return

    if not os.path.exists(TASKS_CSV_PATH):
        raise RuntimeError(
            f"Файл с заданиями не найден: {TASKS_CSV_PATH}"
        )

    with open(TASKS_CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=DELIMITER)

        if not reader.fieldnames or "task" not in reader.fieldnames:
            raise RuntimeError(
                "CSV-файл заданий должен содержать колонку 'task'"
            )

        _tasks_cache = [
            row for row in reader
            if row.get("task") and row["task"].strip()
        ]

    if not _tasks_cache:
        raise RuntimeError("CSV-файл с заданиями пуст")

# ============================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ
# ============================================================

def generate_task() -> str:
    global _last_task_id

    _load_tasks_from_csv()

    available_tasks = _tasks_cache

    if _last_task_id is not None and "id" in _tasks_cache[0]:
        filtered = [
            t for t in _tasks_cache
            if t.get("id") != _last_task_id
        ]
        if filtered:
            available_tasks = filtered

    task = random.choice(available_tasks)

    if "id" in task:
        _last_task_id = task.get("id")

    return task["task"]
