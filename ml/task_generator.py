import csv
import random
import os
from typing import Dict, List

# ============================================================
# НАСТРОЙКИ
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_CSV_PATH = os.path.join(BASE_DIR, "..", "data", "tasks_300.csv")

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
        raise RuntimeError(f"Файл не найден: {TASKS_CSV_PATH}")

    with open(TASKS_CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                _tasks_cache.append(row[0].strip())

    if not _tasks_cache:
        raise RuntimeError("CSV-файл с заданиями пуст")


# ============================================================
# КЛАССИФИКАЦИЯ ТИПА ЗАДАНИЯ
# (позже здесь будет нейросеть)
# ============================================================

def classify_task(task_text: str) -> str:
    """
    ВРЕМЕННО: эвристика.
    В ВКР: заменить на ML-модель.
    """

    t = task_text.lower()

    if "словар" in t and "пары" in t:
        return "dict_items"

    if "словар" in t and "сумм" in t:
        return "dict_sum"

    if "строк" in t and "верхн" in t:
        return "strings_upper"

    if "строк" in t and "длин" in t:
        return "strings_length"

    if "список чисел" in t and "сумм" in t:
        return "list_sum"

    if "список чисел" in t and "чётн" in t:
        return "list_even"

    if "разверните список" in t:
        return "list_reverse"

    # fallback
    return "unknown"


# ============================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ
# ============================================================

def generate_task() -> Dict[str, str]:
    _load_tasks()

    task_text = random.choice(_tasks_cache)
    task_type = classify_task(task_text)

    return {
        "task_text": task_text,
        "task_type": task_type
    }
