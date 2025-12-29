import csv
import random
import os
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TASKS_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "tasks_300.csv")

_tasks_cache: List[str] = []

TASK_TYPE_MAP = {
    1: "list_sum",
    2: "list_count_negative",
    3: "strings_length",
    4: "list_min_max",
    5: "list_even",
    6: "strings_filter_a",
    7: "dict_avg",
    8: "list_product",
    9: "strings_longest",
    10: "list_contains_zero",
    11: "list_sort_asc",
    12: "list_sort_desc",
    13: "strings_join",
    14: "dict_keys",
    15: "dict_values",
    16: "list_count_gt",
    17: "list_index_max",
    18: "strings_lower",
    19: "list_square",
    20: "list_second_max",
    25: "text_count_chars",
    26: "text_count_words",
    40: "strings_upper",
    43: "text_remove_spaces",
    47: "list_reverse",
    51: "list_unique",
    59: "text_replace_vowels",
    71: "list_palindrome",
    87: "text_replace_spaces",
    100: "text_word_lengths",
    101: "list_sum_positive",
    231: "strings_upper",
    234: "text_replace_spaces",
    241: "text_palindrome",
}

def _load_tasks():
    if _tasks_cache:
        return

    with open(TASKS_CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                _tasks_cache.append(row[0].strip())

def generate_task() -> Dict[str, str]:
    _load_tasks()

    task_text = random.choice(_tasks_cache)

    task_id = None
    for token in task_text.replace(".", " ").split():
        if token.isdigit():
            task_id = int(token)
            break

    task_type = TASK_TYPE_MAP.get(task_id, "unsupported")

    return {
        "task_text": task_text,
        "task_type": task_type
    }
