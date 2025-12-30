# checkers.py
# Автор: Федотова Анастасия Алексеевна
# Тема ВКР:
# «Автоматическая генерация и проверка учебных заданий по языку программирования Python
#  с помощью нейронных сетей (на примере ЧОУ ВО „Московский университет имени С.Ю. Витте“)»

import ast
import random
import copy

# ======================================================
# AST-БЕЗОПАСНОСТЬ
# ======================================================

FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.Try,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
    ast.ClassDef,
)

def ast_security_check(code: str):
    """
    Проверяет код пользователя на наличие запрещённых конструкций.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError("В коде использованы запрещённые конструкции")


# ======================================================
# ВЫПОЛНЕНИЕ КОДА ПОЛЬЗОВАТЕЛЯ
# ======================================================

def run_user_code(code: str, env: dict):
    """
    Выполняет код пользователя в изолированной среде.
    Ожидается, что результат будет сохранён в переменной result.
    """
    exec(code, {}, env)
    return env.get("result")


# ======================================================
# ОПИСАНИЯ ПОДДЕРЖИВАЕМЫХ ТИПОВ ЗАДАНИЙ
# ======================================================

TASK_DEFINITIONS = {

    # ---------- СПИСКИ ЧИСЕЛ ----------

    "list_sum": {
        "input": "list[int]",
        "generator": lambda: [random.randint(-10, 10) for _ in range(6)],
        "reference": lambda data: sum(data),
        "vars": ["data", "numbers"]
    },

    "list_filter": {
        "input": "list[int]",
        "generator": lambda: [random.randint(-10, 10) for _ in range(6)],
        "reference": lambda data: [x for x in data if x % 2 == 0],
        "vars": ["data", "numbers"]
    },

    "list_transform": {
        "input": "list[int]",
        "generator": lambda: [random.randint(1, 9) for _ in range(5)],
        "reference": lambda data: [x * x for x in data],
        "vars": ["data", "numbers"]
    },

    "list_reverse": {
        "input": "list[int]",
        "generator": lambda: [random.randint(1, 9) for _ in range(5)],
        "reference": lambda data: data[::-1],
        "vars": ["data", "numbers"]
    },

    # ---------- СТРОКИ ----------

    "text_count": {
        "input": "text",
        "generator": lambda: "Python — это язык программирования",
        "reference": lambda text: len(text.split()),
        "vars": ["text"]
    },

    "text_replace": {
        "input": "text",
        "generator": lambda: "Привет мир Python",
        "reference": lambda text: text.replace(" ", ""),
        "vars": ["text"]
    },

    # ---------- СЛОВАРИ ----------

    "dict_sum": {
        "input": "dict",
        "generator": lambda: {"a": 3, "b": 7, "c": 10},
        "reference": lambda d: sum(d.values()),
        "vars": ["data"]
    },

    "dict_avg": {
        "input": "dict",
        "generator": lambda: {"Ann": 80, "Bob": 95, "Kate": 70},
        "reference": lambda d: sum(d.values()) / len(d),
        "vars": ["data"]
    },
}


# ======================================================
# ОСНОВНАЯ ФУНКЦИЯ ПРОВЕРКИ
# ======================================================

def check_solution(task_type: str, user_code: str):
    """
    Проверяет решение пользователя для заданного типа задания.

    Возвращает:
        (True, сообщение)  — если решение верное
        (False, сообщение) — если решение неверное
    """

    if task_type not in TASK_DEFINITIONS:
        return False, f"Тип задания '{task_type}' не поддерживается системой"

    if "result" not in user_code:
        return False, "В решении должна быть переменная result"

    # Проверка синтаксиса и безопасности
    try:
        ast_security_check(user_code)
        ast.parse(user_code)
    except Exception as e:
        return False, f"Ошибка в коде: {e}"

    task = TASK_DEFINITIONS[task_type]

    # Несколько прогонов для надёжности
    for _ in range(5):
        test_input = task["generator"]()
        expected = task["reference"](copy.deepcopy(test_input))

        env = {}

        # Передаём входные данные пользователю
        for var in task["vars"]:
            env[var] = copy.deepcopy(test_input)

        try:
            user_result = run_user_code(user_code, env)
        except Exception as e:
            return False, f"Ошибка выполнения: {e}"

        if user_result != expected:
            return False, (
                f"Неверный результат.\n"
                f"Ожидалось: {expected}\n"
                f"Получено: {user_result}"
            )

    return True, "Решение верное"
