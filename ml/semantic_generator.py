# ml/semantic_generator.py

import random
import math
import string
import uuid

# ============================================================
# ВНУТРЕННЕЕ ХРАНИЛИЩЕ АКТИВНЫХ ЗАДАНИЙ
# ============================================================

TASK_STORE = {}

# ============================================================
# БАЗОВЫЕ УТИЛИТЫ
# ============================================================

def rand_list_int(min_len=5, max_len=10, a=-10, b=10):
    return [random.randint(a, b) for _ in range(random.randint(min_len, max_len))]


def rand_list_str(min_len=4, max_len=8):
    return [
        "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(3, 7)))
        for _ in range(random.randint(min_len, max_len))
    ]


def rand_dict_int(min_len=5, max_len=8):
    values = rand_list_int(min_len, max_len, 1, 20)
    return {chr(97 + i): v for i, v in enumerate(values)}

# ============================================================
# СЕМАНТИЧЕСКИЕ ГЕНЕРАТОРЫ
# ============================================================

def task_list_sum():
    numbers = rand_list_int()
    return (
        "Дан список чисел numbers.\n"
        "Найдите сумму элементов.\n\n"
        f"numbers = {numbers}",
        {"numbers": numbers},
        lambda env: sum(env["numbers"])
    )


def task_list_positive_sum():
    numbers = rand_list_int()
    return (
        "Дан список чисел numbers.\n"
        "Найдите сумму положительных элементов.\n\n"
        f"numbers = {numbers}",
        {"numbers": numbers},
        lambda env: sum(x for x in env["numbers"] if x > 0)
    )


def task_list_reverse():
    numbers = rand_list_int(5, 8, 1, 9)
    return (
        "Дан список чисел numbers.\n"
        "Разверните список.\n\n"
        f"numbers = {numbers}",
        {"numbers": numbers},
        lambda env: env["numbers"][::-1]
    )


def task_strings_upper():
    words = rand_list_str()
    return (
        "Дан список строк words.\n"
        "Преобразуйте все строки в верхний регистр.\n\n"
        f"words = {words}",
        {"words": words},
        lambda env: [w.upper() for w in env["words"]]
    )


def task_dict_std():
    data = rand_dict_int()

    def reference(env):
        values = list(env["data"].values())
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    return (
        "Дан словарь data.\n"
        "Найдите стандартное отклонение значений.\n\n"
        f"data = {data}",
        {"data": data},
        reference
    )

# ============================================================
# РЕЕСТР СЕМАНТИКИ
# ============================================================

SEMANTIC_TASKS = [
    task_list_sum,
    task_list_positive_sum,
    task_list_reverse,
    task_strings_upper,
    task_dict_std,
]

# ============================================================
# ПУБЛИЧНЫЙ API
# ============================================================

def generate_task():
    """
    Возвращает ТОЛЬКО ТЕКСТ задания (str).
    Всё остальное хранится на сервере.
    """
    task_id = str(uuid.uuid4())
    task_fn = random.choice(SEMANTIC_TASKS)

    text, input_data, reference = task_fn()

    TASK_STORE[task_id] = {
        "input": input_data,
        "reference": reference
    }

    return f"{text}\n\n# task_id: {task_id}"
