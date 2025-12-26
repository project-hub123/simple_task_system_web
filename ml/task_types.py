import ast
import re
from functools import reduce
import operator

# --- Универсальные утилиты для извлечения данных из задания ---

def extract_list(text):
    # Ищет в условии "список чисел", "список строк" и т.д.
    match = re.search(r'=\s*(\[[^\]]+\])', text)
    return ast.literal_eval(match.group(1)) if match else None

def extract_dict(text):
    match = re.search(r'=\s*({[^}]+})', text)
    return ast.literal_eval(match.group(1)) if match else None

def extract_value(text, pattern):
    match = re.search(pattern, text)
    return match.group(1) if match else None

# --- ОПИСАНИЕ ТИПОВ ЗАДАЧ ---

TASK_TYPES = [
    {
        "keywords": ["сумма положительных"],
        "parser": extract_list,
        "checker": lambda data: sum(x for x in data if x > 0),
    },
    {
        "keywords": ["количество отрицательных"],
        "parser": extract_list,
        "checker": lambda data: len([x for x in data if x < 0]),
    },
    {
        "keywords": ["верхний регистр"],
        "parser": extract_list,
        "checker": lambda data: [str(x).upper() for x in data],
    },
    {
        "keywords": ["длину каждой строки"],
        "parser": extract_list,
        "checker": lambda data: [len(str(x)) for x in data],
    },
    {
        "keywords": ["максимальное и минимальное"],
        "parser": extract_list,
        "checker": lambda data: (max(data), min(data)),
    },
    {
        "keywords": ["чётные элементы"],
        "parser": extract_list,
        "checker": lambda data: [x for x in data if x % 2 == 0],
    },
    {
        "keywords": ["среднее значение"],
        "parser": extract_dict,
        "checker": lambda data: sum(data.values()) / len(data),
    },
    {
        "keywords": ["произведение всех элементов"],
        "parser": extract_list,
        "checker": lambda data: reduce(operator.mul, data, 1),
    },
    {
        "keywords": ["самую длинную строку"],
        "parser": extract_list,
        "checker": lambda data: max(data, key=len),
    },
    {
        "keywords": ["есть ли в нём ноль"],
        "parser": extract_list,
        "checker": lambda data: 0 in data,
    },
    # ... Добавляй любые шаблоны!
]
