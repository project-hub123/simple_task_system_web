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
    Ожидаемые колонки:
        task_text
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
# ГЕНЕРАЦИЯ ВХОДНЫХ ДАННЫХ ПО ТИПУ ЗАДАНИЯ
# ======================================================

def generate_input_data(task_type: str) -> str:
    """
    Генерирует входные данные в виде СТРОКИ
    (для последующего ast.literal_eval).
    """

    # ---------- ТЕКСТ ----------
    if task_type.startswith("text"):
        texts = [
            "Python это язык программирования",
            "Анализ данных и машинное обучение",
            "Программирование требует практики",
            "Автоматическая проверка заданий",
            "Разработка программных систем"
        ]
        return random.choice(texts)

    # ---------- СПИСКИ ЧИСЕЛ ----------
    if task_type in {
        "list_sum",
        "list_filter",
        "list_transform",
        "list_reverse"
    }:
        data = [random.randint(-5, 10) for _ in range(6)]
        return str(data)

    # ---------- СПИСКИ СТРОК ----------
    if task_type in {
        "list_unique_strings",
        "list_strings_unique",
        "list_strings_count"
    }:
        data = [
            "яблоко",
            "груша",
            "яблоко",
            "апельсин",
            "груша"
        ]
        return str(data)

    # ---------- СЛОВАРИ ----------
    if task_type in {
        "dict_sum",
        "dict_avg",
        "dict_prices_increase"
    }:
        data = {
            "Хлеб": 50,
            "Молоко": 80,
            "Сыр": 300
        }
        return str(data)

    # ---------- ЗАПАСНОЙ ВАРИАНТ ----------
    return ""


# ======================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ
# ======================================================

def generate_task(use_ml: bool = True) -> Dict[str, str]:
    """
    Выбирает случайное задание и определяет его тип
    с помощью ML-модели (или резервной логики).

    Возвращает:
    {
        "task_text": "...",
        "task_type": "...",
        "input_data": "..."
    }
    """

    load_tasks()

    task = random.choice(_tasks_cache)
    task_text = task["task_text"]

    # Определяем тип задания
    if use_ml:
        task_type = classify_task(task_text)
    else:
        # Резервная логика (на всякий случай)
        task_type = "text_count"

    # Генерируем входные данные
    input_data = generate_input_data(task_type)

    return {
        "task_text": task_text,
        "task_type": task_type,
        "input_data": input_data
    }


# ======================================================
# ЛОКАЛЬНЫЙ ТЕСТ
# ======================================================

if __name__ == "__main__":
    for _ in range(5):
        task = generate_task()
        print("Задание:")
        print(task["task_text"])
        print("Тип:", task["task_type"])
        print("Данные:", task["input_data"])
        print("-" * 40)
