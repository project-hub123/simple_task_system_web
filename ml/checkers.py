import ast
import string

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
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError("В коде использованы запрещённые конструкции")


# ======================================================
# ВЫПОЛНЕНИЕ КОДА ПОЛЬЗОВАТЕЛЯ
# ======================================================

def run_user_code(code: str, env: dict):
    exec(code, {}, env)
    return env.get("result")


# ======================================================
# ЭТАЛОННЫЕ ВЫЧИСЛЕНИЯ
# ======================================================

def calculate_reference(task_type: str, env: dict):

    text = env.get("text")
    data = env.get("data")

    # ---------- TEXT ----------
    if task_type == "text_words":
        return len(text.split())

    if task_type == "text_chars":
        return len(text)

    if task_type == "text_chars_no_space":
        return len(text.replace(" ", ""))

    if task_type == "text_remove_spaces":
        return text.replace(" ", "")

    if task_type == "text_longest_word":
        return max(text.split(), key=len)

    if task_type == "text_shortest_word":
        return min(text.split(), key=len)

    if task_type == "text_sentence_count":
        count = sum(1 for c in text if c in ".!?")
        return count if count > 0 else 1

    if task_type == "text_remove_punct":
        return "".join(c for c in text if c not in string.punctuation + "—")

    if task_type == "text_vowel_count":
        vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
        return sum(1 for c in text if c in vowels)

    # ---------- LIST (numbers) ----------
    if task_type == "list_sum":
        return sum(data)

    if task_type == "list_sum_positive":
        return sum(x for x in data if x > 0)

    if task_type == "list_even":
        return [x for x in data if x % 2 == 0]

    if task_type == "list_square":
        return [x * x for x in data]

    if task_type == "list_max":
        return max(data)

    if task_type == "list_min":
        return min(data)

    if task_type == "list_diff_max_min":
        return max(data) - min(data)

    if task_type == "list_is_sorted":
        return data == sorted(data)

    if task_type == "list_median":
        s = sorted(data)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2

    if task_type == "list_top3":
        return sorted(data, reverse=True)[:3]

    if task_type == "list_max_adjacent_sum":
        return max(data[i] + data[i + 1] for i in range(len(data) - 1))

    if task_type == "list_above_avg":
        avg = sum(data) / len(data)
        return [x for x in data if x > avg]

    if task_type == "list_unique_keep_order":
        seen = []
        for x in data:
            if x not in seen:
                seen.append(x)
        return seen

    if task_type == "list_split_even_odd":
        return {
            "even": [x for x in data if x % 2 == 0],
            "odd": [x for x in data if x % 2 != 0]
        }

    if task_type == "list_sum_even_index":
        return sum(data[i] for i in range(0, len(data), 2))

    if task_type == "list_count_even_odd":
        return {
            "even": sum(1 for x in data if x % 2 == 0),
            "odd": sum(1 for x in data if x % 2 != 0)
        }

    if task_type == "list_index_min":
        return data.index(min(data))

    # ---------- LIST (strings) ----------
    if task_type == "list_unique_strings":
        return len(set(data))

    if task_type == "list_sort_strings":
        return sorted(data)

    if task_type == "list_longest_string":
        return max(data, key=len)

    if task_type == "list_join_comma":
        return ",".join(data)

    if task_type == "list_sort_by_length":
        return sorted(data, key=len)

    if task_type == "list_strings_alpha":
        return [s for s in data if s.isalpha()]

    if task_type == "list_strings_capital":
        return [s for s in data if s and s[0].isupper()]

    if task_type == "list_filter_len_gt_3":
        return [s for s in data if len(s) > 3]

    # ---------- DICT ----------
    if task_type == "dict_sum":
        return sum(data.values())

    if task_type == "dict_avg":
        return sum(data.values()) / len(data) if data else 0

    if task_type == "dict_count":
        return len(data)

    if task_type == "dict_swap":
        return {v: k for k, v in data.items()}

    if task_type == "dict_key_max":
        return max(data, key=data.get)

    if task_type == "dict_values_list":
        return list(data.values())

    raise ValueError(f"Тип задания '{task_type}' не поддерживается")


# ======================================================
# ОСНОВНАЯ ФУНКЦИЯ ПРОВЕРКИ
# ======================================================

def check_solution(task_type: str, user_code: str, input_data: str):

    if "result" not in user_code:
        return False, "В решении должна быть переменная result"

    try:
        ast_security_check(user_code)
        ast.parse(user_code)
    except Exception as e:
        return False, f"Ошибка в коде: {e}"

    env = {}

    try:
        if task_type.startswith("text"):
            env["text"] = input_data
        else:
            env["data"] = ast.literal_eval(input_data)
    except Exception:
        return False, "Ошибка разбора входных данных"

    try:
        expected = calculate_reference(task_type, env)
    except Exception as e:
        return False, f"Ошибка эталонного расчёта: {e}"

    try:
        user_result = run_user_code(user_code, env)
    except Exception as e:
        return False, f"Ошибка выполнения: {e}"

    if user_result != expected:
        return False, (
            "Неверный результат.\n"
            f"Входные данные: {input_data}\n"
            f"Ожидалось: {expected}\n"
            f"Получено: {user_result}"
        )

    return True, "Решение верное"
