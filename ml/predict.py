# ml/predict.py

from .logic_checker import check


def predict(task_text: str, solution_text: str) -> str:
    ok, msg = check(task_text, solution_text)

    if ok:
        return "✅ " + msg
    else:
        return "❌ " + msg
