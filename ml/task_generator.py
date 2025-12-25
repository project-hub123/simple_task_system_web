import csv
import random
import os
from typing import List, Optional

# ============================================================
# НАСТРОЙКИ
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_CSV_PATH = os.path.join(BASE_DIR, "..", "data", "tasks_300.csv")

# ============================================================
# ВНУТРЕННЕЕ ХРАНИЛИЩЕ
# ============================================================

_tasks_cache: List[str] = []
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
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            # Берём ВСЮ строку как одно задание
            task = ",".join(cell.strip() for cell in row).strip()

            if task:
                _tasks_cache.append(task)

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
        filtered = [t for t in _tasks_cache if t != _last_task]
        if filtered:
            available = filtered

    task = random.choice(available)
    _last_task = task

    return task
