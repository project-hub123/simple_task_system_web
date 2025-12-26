import ast

def check_solution(task_text, user_code):
    """
    Логическая (ручная) проверка для задач с известным эталонным ответом.
    task_text: текст задания (например, 'Преобразуйте все строки списка в верхний регистр.')
    user_code: код пользователя (как строка)
    """
    # 1. Проверка синтаксиса
    try:
        ast.parse(user_code)
    except SyntaxError as e:
        return False, f"Синтаксическая ошибка: {e}"

    # 2. Выполнение кода
    user_globals = {}
    try:
        exec(user_code, user_globals)
    except Exception as e:
        return False, f"Ошибка выполнения: {e}"

    # 3. Получение результата
    user_result = user_globals.get('result', None)

    # 4. Сравнение с эталоном для каждой задачи (можно добавлять условия)
    if "строки в верхний регистр" in task_text or "uppercase" in task_text.lower():
        expected = ['PYTHON', 'CODE', 'ANALYSIS']
        if user_result == expected:
            return True, "✅ Всё верно!"
        else:
            return False, f"❌ Ошибка: ожидалось {expected}, получено {user_result}"

    # TODO: добавлять сюда другие условия для других типов задач
    # Например:
    # if "сумма положительных элементов" in task_text:
    #     expected = 42
    #     if user_result == expected:
    #         return True, "✅ Всё верно!"
    #     else:
    #         return False, f"❌ Ошибка: ожидалось {expected}, получено {user_result}"

    # Если для этой задачи нет ручного чекера
    return None, "⚠ Нет логической проверки для этой задачи"
