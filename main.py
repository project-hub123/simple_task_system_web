"""
main.py
Запуск обучения и оценки моделей машинного обучения
ВКР: Система автоматической генерации и проверки заданий по Python
Автор: <Федотова А.А.>
"""

from ml.train import train_model_v1, train_model_v2
from ml.evaluate import evaluate_model

def main():
    print("=== Запуск обучения моделей ===")

    model1, acc1 = train_model_v1()
    print(f"Model v1 accuracy: {acc1:.2f}")

    model2, acc2 = train_model_v2()
    print(f"Model v2 accuracy: {acc2:.2f}")

    print("=== Оценка завершена ===")

if __name__ == "__main__":
    main()
