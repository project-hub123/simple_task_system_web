# ml/predict.py

from .checkers import check_solution


def predict(task: dict, solution_text: str) -> str:
    """
    Основная точка проверки решения.

    task — словарь с ключами:
        - task_text
        - task_type
    solution_text — код пользователя
    """

    # Явно берём тип задания
    task_type = task.get("task_type")

    if not task_type:
        return "⚠ Тип задания не определён"

    try:
        ok, msg = check_solution(task_type, solution_text)
    except Exception as e:
        return f"⚠ Ошибка проверки: {e}"

    if ok is True:
        return "✅ " + msg

    if ok is False:
        return "❌ " + msg

    return "⚠ " + msg
