import ast
from .task_types import TASK_TYPES

def safe_exec(user_code: str):
    env = {}
    exec(user_code, {}, env)
    return env

def check(task_text: str, user_code: str):
    # 1. Синтаксис
    try:
        ast.parse(user_code)
    except SyntaxError as e:
        return False, f"Синтаксическая ошибка: {e}"

    # 2. Выполнение
    try:
        env = safe_exec(user_code)
    except Exception as e:
        return False, f"Ошибка выполнения: {e}"

    if "result" not in env:
        return False, "В коде должна быть переменная result"

    user_result = env["result"]

    # 3. Автоматический поиск типа задачи и вычисление эталона
    for typ in TASK_TYPES:
        if all(kw in task_text for kw in typ["keywords"]):
            data = typ["parser"](task_text)
            if data is None:
                return False, "Не удалось извлечь данные из условия"
            expected = typ["checker"](data)
            if user_result == expected:
                return True, "Решение верное"
            else:
                return False, f"Ожидалось {expected}, получено {user_result}"

    return None, "Тип задачи не поддерживается (добавьте шаблон в TASK_TYPES)"
