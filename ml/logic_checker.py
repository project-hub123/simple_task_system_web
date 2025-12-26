import ast
import copy
import random


# ------------------ ВСПОМОГАТЕЛЬНЫЕ ------------------

def parse_ast(code: str):
    return ast.parse(code)


def extract_assignments(code: str):
    """
    Извлекает ВСЕ присваивания (numbers, lst, arr, a, b, i, j и т.д.)
    """
    tree = ast.parse(code)
    env = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                src = ast.get_source_segment(code, node)
                exec(src, {}, env)
            except Exception:
                pass
    return env


def run_code(code: str, env: dict):
    exec(code, {}, env)
    return env


# ------------------ ГЕНЕРАТОР ТЕСТОВ ------------------

def generate_test_list():
    return [random.randint(-10, 10) for _ in range(6)]


def generate_test_strings():
    return ["ab", "Python", "xy", "Code"]


# ------------------ ОСНОВНАЯ ПРОВЕРКА ------------------

def check(task_text: str, user_code: str):
    # 1. Синтаксис
    try:
        parse_ast(user_code)
    except SyntaxError as e:
        return False, f"Синтаксическая ошибка: {e}"

    # 2. Извлекаем исходные данные студента
    base_env = extract_assignments(user_code)

    # 3. Проверка наличия result
    if "result" not in user_code:
        return False, "В коде должна быть переменная result"

    # ================= СПИСОЧНЫЕ ЗАДАЧИ =================

    if "список чисел" in task_text.lower():
        test_list = generate_test_list()
        env = copy.deepcopy(base_env)

        # пытаемся угадать имя списка
        list_var = None
        for name in ["numbers", "lst", "arr", "a"]:
            if name in env:
                list_var = name
                break
        if not list_var:
            return False, "Не найден список чисел"

        env[list_var] = test_list

        try:
            run_code(user_code, env)
        except Exception as e:
            return False, f"Ошибка выполнения: {e}"

        user_result = env.get("result")

        # --- РАЗВОРОТ ---
        if "разверните" in task_text.lower():
            expected = list(reversed(test_list))
            return (user_result == expected,
                    f"Ожидалось {expected}, получено {user_result}")

        # --- СУММА ---
        if "сумм" in task_text.lower():
            expected = sum(test_list)
            return (user_result == expected,
                    f"Ожидалось {expected}, получено {user_result}")

        # --- ЧЁТНЫЕ ---
        if "чётн" in task_text.lower():
            expected = [x for x in test_list if x % 2 == 0]
            return (user_result == expected,
                    f"Ожидалось {expected}, получено {user_result}")

        # --- МАКС / МИН ---
        if "максим" in task_text.lower():
            expected = max(test_list)
            return (user_result == expected,
                    f"Ожидалось {expected}, получено {user_result}")

        if "миним" in task_text.lower():
            expected = min(test_list)
            return (user_result == expected,
                    f"Ожидалось {expected}, получено {user_result}")

    # ================= СТРОКОВЫЕ ЗАДАЧИ =================

    if "список строк" in task_text.lower():
        test_strings = generate_test_strings()
        env = copy.deepcopy(base_env)

        list_var = None
        for name in ["strings", "lst", "arr", "a"]:
            if name in env:
                list_var = name
                break
        if not list_var:
            return False, "Не найден список строк"

        env[list_var] = test_strings

        try:
            run_code(user_code, env)
        except Exception as e:
            return False, f"Ошибка выполнения: {e}"

        user_result = env.get("result")

        if "верхний регистр" in task_text.lower():
            expected = [s.upper() for s in test_strings]
            return (user_result == expected,
                    f"Ожидалось {expected}, получено {user_result}")

        if "длина" in task_text.lower():
            expected = [len(s) for s in test_strings]
            return (user_result == expected,
                    f"Ожидалось {expected}, получено {user_result}")

    # ================= ИНДЕКСЫ =================

    if "между двумя индексами" in task_text.lower():
        test_list = [1, 2, 3, 4, 5, 6]
        env = copy.deepcopy(base_env)
        env["numbers"] = test_list
        env["i"] = 1
        env["j"] = 4

        try:
            run_code(user_code, env)
        except Exception as e:
            return False, f"Ошибка выполнения: {e}"

        expected = sum(test_list[1:5])
        user_result = env.get("result")
        return (user_result == expected,
                f"Ожидалось {expected}, получено {user_result}")

    # ================= ПО УМОЛЧАНИЮ =================

    return False, "Тип задачи не поддерживается логическим чекером"
