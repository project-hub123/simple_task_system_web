# task_types.py
# Описание типов заданий (метаданные)
# Используется генератором, ML-классификатором и checker'ом

TASK_TYPES = {
    # ===== СПИСКИ ЧИСЕЛ =====

    "list_sum": {
        "input": "list[int]",
        "description": "Сумма элементов списка",
    },

    "list_positive_sum": {
        "input": "list[int]",
        "description": "Сумма положительных элементов",
    },

    "list_negative_count": {
        "input": "list[int]",
        "description": "Количество отрицательных элементов",
    },

    "list_even": {
        "input": "list[int]",
        "description": "Чётные элементы списка",
    },

    "list_product": {
        "input": "list[int]",
        "description": "Произведение элементов списка",
    },

    "list_reverse": {
        "input": "list[int]",
        "description": "Разворот списка",
    },

    # ===== СПИСКИ СТРОК =====

    "strings_upper": {
        "input": "list[str]",
        "description": "Преобразование строк в верхний регистр",
    },

    "strings_length": {
        "input": "list[str]",
        "description": "Длина каждой строки",
    },

    "strings_longest": {
        "input": "list[str]",
        "description": "Самая длинная строка",
    },

    # ===== СЛОВАРИ =====

    "dict_sum": {
        "input": "dict[str,int]",
        "description": "Сумма значений словаря",
    },

    "dict_average": {
        "input": "dict[str,int]",
        "description": "Среднее значение словаря",
    },

    "dict_items": {
        "input": "dict[str,int]",
        "description": "Список пар (ключ, значение)",
    },
}
