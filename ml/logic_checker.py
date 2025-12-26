import ast
import copy
import random
import re
from statistics import mean, median


# -------------------- ВСПОМОГАТЕЛЬНЫЕ --------------------

def safe_exec(code: str, env: dict):
    exec(code, {}, env)
    return env


def extract_assignments(code: str):
    tree = ast.parse(code)
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                exec(ast.get_source_segment(code, node), {}, env)
            except Exception:
                pass
    return env


def has_result(code: str):
    return "result" in code


# -------------------- ТЕСТОВЫЕ ДАННЫЕ --------------------

TEST_LIST_NUM = [3, -1, 5, 0, -7, 9]
TEST_LIST_STR = ["Apple", "banana", "PEAR", "cat", "dog123"]
TEST_DICT = {"a": 2, "b": 5, "c": 10}
TEST_TEXT = "Anna has 2 apples and 3 bananas."

# -------------------- ОСНОВНАЯ ПРОВЕРКА --------------------

def check(task_text: str, user_code: str):
    task = task_text.lower()

    # 1. Синтаксис
    try:
        ast.parse(user_code)
    except SyntaxError as e:
        return False, f"Синтаксическая ошибка: {e}"

    if not has_result(user_code):
        return False, "В коде должна быть переменная result"

    base_env = extract_assignments(user_code)

    # ---------- СПИСКИ ЧИСЕЛ ----------
    if "список чисел" in task:
        env = copy.deepcopy(base_env)
        env["data"] = TEST_LIST_NUM
        env["numbers"] = TEST_LIST_NUM

        try:
            safe_exec(user_code, env)
        except Exception as e:
            return False, f"Ошибка выполнения: {e}"

        r = env.get("result")

        if "сумм" in task:
            return (r == sum(TEST_LIST_NUM),
                    f"Ожидалось {sum(TEST_LIST_NUM)}, получено {r}")

        if "максим" in task:
            return (r == max(TEST_LIST_NUM),
                    f"Ожидалось {max(TEST_LIST_NUM)}, получено {r}")

        if "миним" in task:
            return (r == min(TEST_LIST_NUM),
                    f"Ожидалось {min(TEST_LIST_NUM)}, получено {r}")

        if "чётн" in task:
            exp = [x for x in TEST_LIST_NUM if x % 2 == 0]
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "разверните" in task:
            exp = TEST_LIST_NUM[::-1]
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "среднее" in task:
            exp = mean(TEST_LIST_NUM)
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "медиан" in task:
            exp = median(TEST_LIST_NUM)
            return (r == exp, f"Ожидалось {exp}, получено {r}")

    # ---------- СПИСКИ СТРОК ----------
    if "список строк" in task:
        env = copy.deepcopy(base_env)
        env["words"] = TEST_LIST_STR
        env["strings"] = TEST_LIST_STR

        try:
            safe_exec(user_code, env)
        except Exception as e:
            return False, f"Ошибка выполнения: {e}"

        r = env.get("result")

        if "длин" in task:
            exp = [len(s) for s in TEST_LIST_STR]
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "верхний" in task:
            exp = [s.upper() for s in TEST_LIST_STR]
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "нижний" in task:
            exp = [s.lower() for s in TEST_LIST_STR]
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "букву" in task:
            exp = [s for s in TEST_LIST_STR if "a" in s.lower()]
            return (r == exp, f"Ожидалось {exp}, получено {r}")

    # ---------- СЛОВАРИ ----------
    if "словар" in task:
        env = copy.deepcopy(base_env)
        env["data"] = TEST_DICT
        env["scores"] = TEST_DICT

        try:
            safe_exec(user_code, env)
        except Exception as e:
            return False, f"Ошибка выполнения: {e}"

        r = env.get("result")

        if "сумм" in task:
            exp = sum(TEST_DICT.values())
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "средн" in task:
            exp = mean(TEST_DICT.values())
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "квадрат" in task:
            exp = sum(v*v for v in TEST_DICT.values())
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "ключ" in task and "максим" in task:
            exp = max(TEST_DICT, key=TEST_DICT.get)
            return (r == exp, f"Ожидалось {exp}, получено {r}")

    # ---------- ТЕКСТ ----------
    if "текст" in task:
        env = copy.deepcopy(base_env)
        env["text"] = TEST_TEXT

        try:
            safe_exec(user_code, env)
        except Exception as e:
            return False, f"Ошибка выполнения: {e}"

        r = env.get("result")

        if "слов" in task:
            exp = len(TEST_TEXT.split())
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "символ" in task:
            exp = len(TEST_TEXT.replace(" ", ""))
            return (r == exp, f"Ожидалось {exp}, получено {r}")

        if "палиндром" in task:
            t = TEST_TEXT.lower().replace(" ", "")
            exp = t == t[::-1]
            return (r == exp, f"Ожидалось {exp}, получено {r}")

    return False, "Тип задачи не поддерживается данным чекером"
