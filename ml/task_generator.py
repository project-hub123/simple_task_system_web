import csv
import random
from typing import List, Dict

# ============================================================
# НАСТРОЙКИ
# ============================================================

TASKS_CSV_PATH = "data/tasks_300.csv"
DELIMITER = ";"

# ============================================================
# ВНУТРЕННЕЕ ХРАНИЛИЩЕ
# ============================================================

_tasks_cache: List[Dict[str, str]] = []
_last_task_id: str | None = None

# ============================================================
# ЗАГРУЗКА ЗАДАНИЙ ИЗ CSV
# ============================================================

def _load_tasks_from_csv() -> None:
    """
    Загружает задания из CSV-файла в память.
    CSV формат:
    id ; task
    """
    global _tasks_cache

    if _tasks_cache:
        return  # уже загружены

    try:
        with open(TASKS_CSV_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=DELIMITER)
            _tasks_cache = [row for row in reader if row.get("task")]
    except FileNotFoundError:
        raise RuntimeError(
            f"Файл с заданиями не найден: {TASKS_CSV_PATH}"
        )

    if not _tasks_cache:
        raise RuntimeError("CSV-файл с заданиями пуст")

# ============================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ
# ============================================================

def generate_task() -> str:
    """
    Возвращает случайное учебное задание из CSV.
    Гарантирует, что подряд не вернётся одно и то же задание.
    """

    global _last_task_id

    _load_tasks_from_csv()

    available_tasks = _tasks_cache

    if _last_task_id is not None:
        filtered = [
            t for t in _tasks_cache if t["id"] != _last_task_id
        ]
        if filtered:
            available_tasks = filtered

    task = random.choice(available_tasks)
    _last_task_id = task["id"]

    return task["task"]
