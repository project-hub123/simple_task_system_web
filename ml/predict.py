from .logic_checker import check

def predict(task_text: str, solution_text: str) -> str:
    ok, msg = check(task_text, solution_text)
    if ok is True:
        return "✅ " + msg
    if ok is False:
        return "❌ " + msg
    return "⚠ " + msg
