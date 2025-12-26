# ml/logic_checker.py

import ast
import copy


def split_user_code(code: str):
    """
    Отделяет вычисление result от остального кода
    """
    tree = ast.parse(code)
    before = []
    result_expr = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "result":
                    result_expr = node.value
                    break
            else:
                before.append(node)
        else:
            before.append(node)

    if result_expr is None:
        raise ValueError("В коде должна быть переменная result")

    return before, result_expr


def exec_nodes(nodes):
    env = {}
    module = ast.Module(body=nodes, type_ignores=[])
    compiled = compile(module, "<user_code>", "exec")
    exec(compiled, {}, env)
    return env


def evaluate(expr, env):
    return eval(compile(ast.Expression(expr), "<expr>", "eval"), {}, env)


def check(task_text: str, user_code: str):
    # 1. Синтаксис
    try:
        ast.parse(user_code)
    except SyntaxError as e:
        return False, f"Синтаксическая ошибка: {e}"

    # 2. Разделяем код
    try:
        base_nodes, result_expr = split_user_code(user_code)
    except Exception as e:
        return False, str(e)

    # 3. Выполняем код ДО result
    try:
        env = exec_nodes(base_nodes)
    except Exception as e:
        return False, f"Ошибка выполнения кода: {e}"

    # 4. Эталон: вычисляем result повторно
    try:
        expected = evaluate(result_expr, copy.deepcopy(env))
    except Exception as e:
        return False, f"Ошибка вычисления эталона: {e}"

    # 5. Выполняем полный код пользователя
    try:
        full_env = {}
        exec(user_code, {}, full_env)
    except Exception as e:
        return False, f"Ошибка выполнения кода: {e}"

    user_result = full_env.get("result", None)

    # 6. Сравнение
    if user_result == expected:
        return True, "Решение верное"
    else:
        return False, f"Ожидалось {expected}, получено {user_result}"
