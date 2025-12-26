# ml/logic_checker.py

import ast
import copy
import random
import re
from statistics import mean, median


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
)

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "__import__", "compile", "input"
}


def ast_security_check(code: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError("Запрещённая конструкция в коде")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
                raise ValueError(f"Запрещённая функция: {node.func.id}")


# ======================================================
# ВСПОМОГАТЕЛЬНЫЕ
# ======================================================

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


# ======================================================
# ГЕНЕРАТОРЫ ТЕСТОВЫХ ДАННЫХ
# ======================================================

def gen_list_numbers():
    return [random.randint(-10, 10) for _ in range(6)]


def gen_list_strings():
    return ["Apple", "banana", "Cat", "dog", "py3"]


def gen_dict_numbers():
    return {"a": random.randint(1, 20), "b": random.randint(1, 20), "c": random.randint(1, 20)}


def gen_text():
    return "Anna has 2 apples and 3 bananas"


# ======================================================
# ОСНОВНАЯ ПРОВЕРКА
# ======================================================

def check(task_text: str, user_code: str):
    task = task_text.lower()

    # --- AST и базовые проверки ---
    try:
        ast_security_check(user_code)
        ast.parse(user_code)
    except Exception as e:
        return False, str(e)

    if "result" not in user_code:
        return False, "В коде должна быть переменная result"

    base_env = extract_assignments(user_code)

    # --- несколько тестовых прогонов ---
    for _ in range(5):
        env = copy.deepcopy(base_env)

        # ================= СПИСКИ ЧИСЕЛ =================
        if "список чисел" in task:
            data = gen_list_numbers()
            env["data"] = data
            env["numbers"] = data

            safe_exec(user_code, env)
            r = env.get("result")

            if "сумм" in task:
                if r != sum(data):
                    return False, f"Ожидалось {sum(data)}, получено {r}"

            elif "чётн" in task:
                exp = [x for x in data if x % 2 == 0]
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            elif "разверните" in task:
                exp = data[::-1]
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            elif "максим" in task and "индекс" not in task:
                if r != max(data):
                    return False, f"Ожидалось {max(data)}, получено {r}"

            elif "миним" in task:
                if r != min(data):
                    return False, f"Ожидалось {min(data)}, получено {r}"

            elif "средн" in task:
                if r != mean(data):
                    return False, f"Ожидалось {mean(data)}, получено {r}"

            elif "медиан" in task:
                if r != median(data):
                    return False, f"Ожидалось {median(data)}, получено {r}"

            elif "между двумя индексами" in task:
                i, j = 1, len(data) - 2
                if r != sum(data[i:j+1]):
                    return False, f"Ожидалось {sum(data[i:j+1])}, получено {r}"

        # ================= СПИСКИ СТРОК =================
        if "список строк" in task:
            words = gen_list_strings()
            env["words"] = words

            safe_exec(user_code, env)
            r = env.get("result")

            if "длин" in task:
                exp = [len(w) for w in words]
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            elif "верхний" in task:
                exp = [w.upper() for w in words]
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            elif "нижний" in task:
                exp = [w.lower() for w in words]
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            elif "букву" in task:
                exp = [w for w in words if "a" in w.lower()]
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

        # ================= СЛОВАРИ =================
        if "словар" in task:
            d = gen_dict_numbers()
            env["data"] = d
            env["scores"] = d

            safe_exec(user_code, env)
            r = env.get("result")

            # 14, 273
            if "ключ" in task and "список пар" not in task:
                exp = list(d.keys())
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            # 15, 42, 232, 282
            elif "значени" in task and "список пар" not in task:
                exp = list(d.values())
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            # 269
            elif "список пар" in task or "пары" in task:
                exp = list(d.items())
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            # 46, 225
            elif "сумм" in task:
                exp = sum(d.values())
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            # 7, 66
            elif "средн" in task:
                exp = mean(d.values())
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            # 142
            elif "квадрат" in task:
                exp = sum(v*v for v in d.values())
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            # 53, 222
            elif "поменяйте" in task:
                exp = {v: k for k, v in d.items()}
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

        # ================= ТЕКСТ =================
        if "текст" in task:
            text = gen_text()
            env["text"] = text

            safe_exec(user_code, env)
            r = env.get("result")

            if "слов" in task:
                exp = len(text.split())
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            elif "символ" in task and "пробел" not in task:
                exp = len(text.replace(" ", ""))
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

            elif "палиндром" in task:
                t = re.sub(r"\W", "", text.lower())
                exp = t == t[::-1]
                if r != exp:
                    return False, f"Ожидалось {exp}, получено {r}"

    return True, "Решение верное"
