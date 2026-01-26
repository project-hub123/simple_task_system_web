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
    Проверка решения пользователя с использованием модели
    машинного обучения для определения типа задания.
    """

    # -------------------------------
    # Проверка входных данных
    # -------------------------------

    if not isinstance(task, dict):
        return "⚠ Ошибка: некорректное описание задания"

    task_text = task.get("task_text", "")
    expected_task_type = task.get("task_type")
    input_data = task.get("input_data")

    if not task_text:
        return "⚠ Ошибка: текст задания отсутствует"

    # -------------------------------
    # Определение типа задания моделью
    # -------------------------------

    try:
        predicted_task_type = _predict_task_type(task_text)
    except Exception as e:
        return f"⚠ Ошибка определения типа задания: {e}"

    # -------------------------------
    # ЖЁСТКАЯ ЗАЩИТА ПО ТИПУ ДАННЫХ
    # -------------------------------

    task_type = predicted_task_type
    type_info = f"Тип задания определён моделью: {predicted_task_type}"

    # если входные данные — список строк,
    # запрещаем любые суммирования
    if isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
        task_type = "list_filter_len_gt_3"
        type_info = (
            "Тип задания определён по структуре входных данных "
            "(список строк)"
        )

    # если тип явно задан в задании и не противоречит данным — используем его
    elif expected_task_type:
        task_type = expected_task_type
        type_info = (
            f"Тип задания взят из описания задания: {expected_task_type}"
        )

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

    prefix = "✅" if ok else "❌"

    return (
        f"{prefix} {message}\n"
        f"{type_info}"
    )
# -------------------------------------------------
# Экспорт функции предсказания типа
# -------------------------------------------------

predict_task_type = _predict_task_type