import random
import math
import string


# ============================================================
# БАЗОВЫЕ УТИЛИТЫ
# ============================================================

def rand_list_int(min_len=5, max_len=10, a=-10, b=10):
    return [random.randint(a, b) for _ in range(random.randint(min_len, max_len))]


def rand_list_str(min_len=4, max_len=8):
    words = []
    for _ in range(random.randint(min_len, max_len)):
        size = random.randint(3, 7)
        words.append("".join(random.choice(string.ascii_lowercase) for _ in range(size)))
    return words


def rand_dict_int(min_len=5, max_len=8):
    values = rand_list_int(min_len, max_len, 1, 20)
    return {chr(97 + i): v for i, v in enumerate(values)}


# ============================================================
# СЕМАНТИЧЕСКИЕ КЛАССЫ ЗАДАНИЙ
# ============================================================

def task_list_sum():
    numbers = rand_list_int()

    return {
        "task_text": "Дан список чисел. Найдите сумму элементов.",
        "input_data": {"numbers": numbers},
        "reference": lambda env: sum(env["numbers"])
    }


def task_list_positive_sum():
    numbers = rand_list_int()

    return {
        "task_text": "Дан список чисел. Найдите сумму положительных элементов.",
        "input_data": {"numbers": numbers},
        "reference": lambda env: sum(x for x in env["numbers"] if x > 0)
    }


def task_list_reverse():
    numbers = rand_list_int(5, 8, 1, 9)

    return {
        "task_text": "Дан список чисел. Разверните список.",
        "input_data": {"numbers": numbers},
        "reference": lambda env: env["numbers"][::-1]
    }


def task_list_even():
    numbers = rand_list_int()

    return {
        "task_text": "Дан список чисел. Выведите чётные элементы.",
        "input_data": {"numbers": numbers},
        "reference": lambda env: [x for x in env["numbers"] if x % 2 == 0]
    }


def task_strings_upper():
    words = rand_list_str()

    return {
        "task_text": "Дан список строк. Преобразуйте все строки в верхний регистр.",
        "input_data": {"words": words},
        "reference": lambda env: [w.upper() for w in env["words"]]
    }


def task_strings_length():
    words = rand_list_str()

    return {
        "task_text": "Дан список строк. Найдите длину каждой строки.",
        "input_data": {"words": words},
        "reference": lambda env: [len(w) for w in env["words"]]
    }


def task_dict_items():
    data = rand_dict_int()

    return {
        "task_text": "Дан словарь. Сформируйте список пар (ключ, значение).",
        "input_data": {"data": data},
        "reference": lambda env: list(env["data"].items())
    }


def task_dict_sum():
    data = rand_dict_int()

    return {
        "task_text": "Дан словарь. Найдите сумму всех значений.",
        "input_data": {"data": data},
        "reference": lambda env: sum(env["data"].values())
    }


def task_dict_std():
    data = rand_dict_int()

    def reference(env):
        values = list(env["data"].values())
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    return {
        "task_text": "Дан словарь. Найдите стандартное отклонение значений.",
        "input_data": {"data": data},
        "reference": reference
    }


# ============================================================
# РЕЕСТР СЕМАНТИЧЕСКИХ ОПЕРАЦИЙ
# ============================================================

SEMANTIC_TASKS = [
    task_list_sum,
    task_list_positive_sum,
    task_list_reverse,
    task_list_even,
    task_strings_upper,
    task_strings_length,
    task_dict_items,
    task_dict_sum,
    task_dict_std,
    # сюда добавляются новые семантические операции
]


# ============================================================
# ПУБЛИЧНЫЙ API
# ============================================================

def generate_task():
    """
    Генерирует произвольное задание из семантического пространства
    """
    task_fn = random.choice(SEMANTIC_TASKS)
    return task_fn()
