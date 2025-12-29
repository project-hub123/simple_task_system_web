# ml/predict.py

from .checkers import check_solution


def predict(task_type: str, solution_text: str) -> str:
    """
    Основная точка проверки решения.

    task_type: тип задания (например: 'dict_items', 'list_reverse', ...)
    solution_text: код пользователя
    """

    ok, msg = check_solution(task_type, solution_text)

    if ok is True:
        return "✅ " + msg
    elif ok is False:
        return "❌ " + msg
    else:
        return "⚠ " + msg
