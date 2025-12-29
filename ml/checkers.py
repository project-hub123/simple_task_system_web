import ast
import copy
import random

# ======================================================
# БЕЗОПАСНОСТЬ AST
# ======================================================

FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.Try,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
)

def ast_security_check(code: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError("Запрещённая конструкция в коде")

# ======================================================
# ВСПОМОГАТЕЛЬНОЕ
# ======================================================

def run_user_code(code: str, env: dict):
    exec(code, {}, env)
    return env.get("result")

# ======================================================
# ТЕКСТОВЫЕ ЗАДАНИЯ
# ======================================================

def check_text_remove_spaces(user_code: str):
    test_input = "Привет мир Python"
    expected = "ПриветмирPython"

    env = {"text": test_input}

    try:
        ast_security_check(user_code)
        ast.parse(user_code)
        exec(user_code, {}, env)
    except Exception as e:
        return False, f"Ошибка выполнения: {e}"

    if "result" not in env:
        return False, "В коде должна быть переменная result"

    if env["result"] == expected:
        return True, "Пробелы удалены корректно"

    return False, f"Ожидалось '{expected}', получено '{env['result']}'"

# ======================================================
# СТРУКТУРИРОВАННЫЕ ЗАДАНИЯ (СПИСКИ, СЛОВАРИ)
# ======================================================

TASK_DEFINITIONS = {
    "list_sum": {
        "input": "list[int]",
        "reference": lambda data: sum(data),
        "generator": lambda: [random.randint(-10, 10) for _ in range(6)],
    },
    "list_even": {
        "input": "list[int]",
        "reference": lambda data: [x for x in data if x % 2 == 0],
        "generator": lambda: [random.randint(-10, 10) for _ in range(6)],
    },
    "list_reverse": {
        "input": "list[int]",
        "reference": lambda data: data[::-1],
        "generator": lambda: [random.randint(-10, 10) for _ in range(6)],
    },
    "strings_upper": {
        "input": "list[str]",
        "reference": lambda data: [s.upper() for s in data],
        "generator": lambda: ["python", "code", "analysis"],
    },
    "strings_length": {
        "input": "list[str]",
        "reference": lambda data: [len(s) for s in data],
        "generator": lambda: ["python", "ai", "ml"],
    },
    "dict_items": {
        "input": "dict",
        "reference": lambda d: list(d.items()),
        "generator": lambda: {"a": 1, "b": 2, "c": 3},
    },
    "dict_sum": {
        "input": "dict",
        "reference": lambda d: sum(d.values()),
        "generator": lambda: {"a": 3, "b": 7, "c": 10},
    },
}

# ======================================================
# ОСНОВНАЯ ПРОВЕРКА
# ======================================================

def check_solution(task_type: str, user_code: str):
    """
    Универсальная автоматическая проверка решения.
    """

    # --- ТЕКСТ ---
    if task_type == "text_remove_spaces":
        return check_text_remove_spaces(user_code)

    # --- СПИСКИ / СЛОВАРИ ---
    if task_type not in TASK_DEFINITIONS:
        return False, "Тип задания не поддерживается системой"

    task = TASK_DEFINITIONS[task_type]

    try:
        ast_security_check(user_code)
        ast.parse(user_code)
    except Exception as e:
        return False, f"Ошибка в коде: {e}"

    if "result" not in user_code:
        return False, "В коде должна быть переменная result"

    for _ in range(5):
        test_data = task["generator"]()
        expected = task["reference"](copy.deepcopy(test_data))

        env = {}

        if task["input"].startswith("list"):
            env["data"] = test_data
            env["numbers"] = test_data
            env["words"] = test_data
        elif task["input"] == "dict":
            env["data"] = test_data

        try:
            user_result = run_user_code(user_code, env)
        except Exception as e:
            return False, f"Ошибка выполнения: {e}"

        if user_result != expected:
            return False, (
                f"Неверно: ожидалось {expected}, "
                f"получено {user_result}"
            )

    return True, "Решение верное"
