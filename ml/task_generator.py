import random
from typing import Dict, Any

# ======================================================
# СПРАВОЧНИК ТИПОВ ЗАДАНИЙ
# ======================================================

TASKS = {
    "list_sum": {
        "description": "Дан список чисел. Найдите сумму элементов списка.",
        "input": lambda: [random.randint(1, 10) for _ in range(6)],
        "solve": lambda data: sum(data)
    },
    "list_even": {
        "description": "Дан список чисел. Найдите количество чётных элементов.",
        "input": lambda: [random.randint(1, 20) for _ in range(8)],
        "solve": lambda data: len([x for x in data if x % 2 == 0])
    },
    "list_sort": {
        "description": "Дан список чисел. Отсортируйте список по возрастанию.",
        "input": lambda: [random.randint(1, 50) for _ in range(7)],
        "solve": lambda data: sorted(data)
    },
    "text_chars": {
        "description": "Дана строка текста. Найдите количество символов без учёта пробелов.",
        "input": lambda: "Анализ данных и машинное обучение",
        "solve": lambda text: len(text.replace(" ", ""))
    },
    "text_words": {
        "description": "Дана строка текста. Найдите количество слов.",
        "input": lambda: "Анализ данных и машинное обучение",
        "solve": lambda text: len(text.split())
    }
}

# ======================================================
# ГЕНЕРАЦИЯ ЗАДАНИЯ
# ======================================================

def generate_task() -> Dict[str, Any]:
    task_type = random.choice(list(TASKS.keys()))
    task = TASKS[task_type]

    input_data = task["input"]()
    expected_result = task["solve"](input_data)

    if isinstance(input_data, list):
        input_text = f"data = Список: {input_data}"
    else:
        input_text = f'data = Строка текста: "{input_data}"'

    return {
        "task_type": task_type,
        "task_text": task["description"],
        "input_data": input_text,
        "expected_result": expected_result
    }

# ======================================================
# ЛОКАЛЬНЫЙ ТЕСТ
# ======================================================

if __name__ == "__main__":
    task = generate_task()
    print("ТИП ЗАДАНИЯ:", task["task_type"])
    print("ЗАДАНИЕ:", task["task_text"])
    print("ВХОДНЫЕ ДАННЫЕ:", task["input_data"])
    print("ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:", task["expected_result"])
