# predict.py
# Автор: Федотова Анастасия Алексеевна
# Тема ВКР:
# «Автоматическая генерация и проверка учебных заданий по языку программирования Python
#  с помощью нейронных сетей (на примере ЧОУ ВО „Московский университет имени С.Ю. Витте“)»

from ml.checkers import check_solution


def predict(task: dict, solution_text: str) -> str:
    """
    Основная точка проверки решения пользователя.

    task — словарь вида:
    {
        "task_text": "...",
        "task_type": "list_sum" | "text_count" | ...
    }

    solution_text — код пользователя (строкой)

    Возвращает строку с результатом проверки.
    """

    if not isinstance(task, dict):
        return "⚠ Ошибка: некорректное описание задания"

    task_type = task.get("task_type")

    if not task_type:
        return "⚠ Ошибка: тип задания не определён"

    try:
        ok, message = check_solution(task_type, solution_text)
    except Exception as e:
        return f"⚠ Ошибка проверки: {e}"

    if ok is True:
        return "✅ " + message
    else:
        return "❌ " + message
