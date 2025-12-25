import csv
import random
import os
from typing import List, Dict, Optional

# ============================================================
# НАСТРОЙКИ
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_CSV_PATH = os.path.join(BASE_DIR, "..", "data", "tasks_300.csv")

# ============================================================
# ВНУТРЕННЕЕ ХРАНИЛИЩЕ
# ============================================================

_tasks_cache: List[Dict[str, str]] = []
_last_task: Optional[str] = None

# ============================================================
# ЗАГРУЗКА ЗАДАНИЙ ИЗ CSV
# ============================================================

def _load_tasks_from_csv() -> None:
    global _tasks_cache

    if _tasks_cache:
        return

    if not os.path.exists(TASKS_CSV_PATH):
        raise RuntimeError(f"Файл не найден: {TASKS_CSV_PATH}")

    with open(TASKS_CSV_PATH, encoding="utf-8-sig") as f:
        sample = f.read(1024)
        f.seek(0)

        # автоопределение разделителя
        delimiter = ";" if ";" in sample else ","

        reader = csv.DictReader(f, delimiter=delimiter)

        if not reader.fieldnames:
            raise RuntimeError("CSV-файл пуст или повреждён")

        # ищем колонку с заданием
        task_field = None
        for name in reader.fieldnames:
            if name.strip().lower() in ("task", "задание"):
                task_field = name
                break

        if not task_field:
            raise RuntimeError(
                f"В CSV нет колонки 'task'. Найдено: {reader.fieldnames}"
            )

        _tasks_cache = [
            {"task": row[task_field].strip()}
            for row in reader
            if row.get(task_field) and row[task_field].strip()
        ]

    if not _tasks_cache:
        raise RuntimeError("CSV-файл с заданиями пуст")

# ============================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ
# ============================================================

def generate_task() -> str:
    global _last_task

    _load_tasks_from_csv()

    available = _tasks_cache

    if _last_task:
        filtered = [t for t in _tasks_cache if t["task"] != _last_task]
        if filtered:
            available = filtered

    task = random.choice(available)["task"]
    _last_task = task

    return task
