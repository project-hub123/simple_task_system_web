from ml.checkers import check_solution
from ml.model_service import load_model, model_exists


# -------------------------------------------------
# КЭШ МОДЕЛИ
# -------------------------------------------------

_MODEL_CACHE = None


def _get_model():
    global _MODEL_CACHE

    if not model_exists():
        raise RuntimeError("Обученная модель не найдена")

    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_model()

    return _MODEL_CACHE


def _predict_task_type(task_text: str) -> str:
    """
    Определение типа задания с помощью обученной модели.
    """
    model_data = _get_model()

    vectorizer = model_data["vectorizer"]
    model = model_data["model"]

    X = vectorizer.transform([task_text])
    return model.predict(X)[0]


# -------------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ
# -------------------------------------------------

def predict(task: dict, solution_text: str) -> str:
    """
    Основная точка проверки решения пользователя.

    task — словарь вида:
    {
        "task_text": "...",
        "task_type": "...",   
        "input_data": "..."
    }

    solution_text — код пользователя (строкой)

    Возвращает строку с результатом проверки.
    """

    # -------------------------------
    # Проверка входных данных
    # -------------------------------

    if not isinstance(task, dict):
        return "⚠ Ошибка: некорректное описание задания"

    task_text = task.get("task_text", "")
    expected_task_type = task.get("task_type")
    input_data = task.get("input_data", "")

    if not task_text:
        return "⚠ Ошибка: текст задания отсутствует"

    # -------------------------------
    # Определение типа задания (ML)
    # -------------------------------

    try:
        predicted_task_type = _predict_task_type(task_text)
    except Exception as e:
        return f"⚠ Ошибка определения типа задания: {e}"

    # -------------------------------
    # Использование task_type ОСМЫСЛЕННО
    # -------------------------------

    if expected_task_type and expected_task_type != predicted_task_type:
        # при расхождении приоритет у модели
        task_type = predicted_task_type
    else:
        task_type = predicted_task_type

    # -------------------------------
    # Проверка решения
    # -------------------------------

    try:
        ok, message = check_solution(
            task_type=task_type,
            user_code=solution_text,
            input_data=input_data
        )
    except Exception as e:
        return f"⚠ Ошибка проверки: {e}"

    # -------------------------------
    # Формирование ответа
    # -------------------------------

    if ok:
        return "✅ " + message
    else:
        return "❌ " + message
