"""
Модуль машинного обучения для интеллектуального сервиса генерации
и проверки заданий по Python.

Используется в веб-приложении Flask и в standalone-обучении моделей.
"""

from .ml_model import (
    load_local_model,
    predict_local_feedback,
    retrain_model,
    evaluate_model,
    get_model_stats
)

from .train import (
    train_model_v1,
    train_model_v2
)

__all__ = [
    "load_local_model",
    "predict_local_feedback",
    "retrain_model",
    "evaluate_model",
    "get_model_stats",
    "train_model_v1",
    "train_model_v2"
]
