import ast

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
# РАЗБОР ВХОДНЫХ ДАННЫХ
# ======================================================

def parse_input(input_data: str):
    """
    Поддерживаем форматы:
    data = Список: [1, 2, 3]
    data = Строка текста: "text"
    """
    if ":" not in input_data:
        raise ValueError("Некорректный формат входных данных")

    _, value = input_data.split(":", 1)
    value = value.strip()

    if value.startswith("["):
        return ast.literal_eval(value)

    if value.startswith('"') or value.startswith("'"):
        return ast.literal_eval(value)

    return value

# ======================================================
# ОСНОВНАЯ ФУНКЦИЯ ПРОВЕРКИ
# ======================================================

def check_solution(
    task_type: str,
    user_code: str,
    input_data: str,
    expected_result
):
    if "result" not in user_code:
        return False, "В решении должна быть переменная result"

    # --- Проверка AST
    try:
        ast_security_check(user_code)
        ast.parse(user_code)
    except Exception as e:
        return False, f"Ошибка в коде: {e}"

    # --- Подготовка окружения
    env = {}
    try:
        parsed_data = parse_input(input_data)
        if isinstance(parsed_data, list):
            env["data"] = parsed_data
        else:
            env["text"] = parsed_data
    except Exception:
        return False, "Ошибка разбора входных данных"

    # --- Выполнение кода пользователя
    try:
        user_result = run_user_code(user_code, env)
    except Exception as e:
        return False, f"Ошибка выполнения: {e}"

    # --- Сравнение
    if user_result != expected_result:
        return False, (
            "Неверный результат.\n"
            f"Входные данные: {input_data}\n"
            f"Ожидалось: {expected_result}\n"
            f"Получено: {user_result}"
        )

    return True, "Решение верное"
