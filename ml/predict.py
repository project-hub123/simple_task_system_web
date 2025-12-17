import joblib

MODEL_PATH = "models/model_v1.pkl"

_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict(solution_text: str) -> str:
    model = load_model()
    result = model.predict([solution_text])[0]

    if result == 1:
        return "Решение корректное. Код соответствует требованиям."
    else:
        return "Обнаружены ошибки или несоответствия в решении."
