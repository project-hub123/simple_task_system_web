# ml/predict.py

from .checkers import check_solution


def predict(task: dict, solution_text: str) -> str:
    """
    Основная точка проверки решения.

    task: словарь с ключами:
        - task_text
        - task_type
    solution_text: код пользователя
    """

    task_type = task.get("task_type", "general")

    try:
        ok, msg = check_solution(task_type, solution_text)
    except Exception as e:
        return f"⚠ Ошибка проверки: {e}"

    if ok is True:
        return "✅ " + msg
    elif ok is False:
        return "❌ " + msg
    else:
        return "⚠ " + msg
