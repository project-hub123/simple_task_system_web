from ml.train import train_model_v1, train_model_v2
from ml_model import load_local_model, evaluate_model

def main():
    print("=" * 50)
    print(" ML SYSTEM START ")
    print("=" * 50)

    # ===== МОДЕЛЬ V1 =====
    print("\n[1] Обучение модели V1 (train / test split)")
    model_v1, acc_v1 = train_model_v1()
    print(f"Accuracy V1: {acc_v1:.3f}")

    if acc_v1 < 0.7:
        print("⚠ ВНИМАНИЕ: Accuracy ниже 70%")

    # ===== МОДЕЛЬ V2 =====
    print("\n[2] Обучение модели V2 (полный датасет)")
    model_v2, acc_v2 = train_model_v2()
    print(f"Accuracy V2: {acc_v2:.3f}")

    # ===== ЗАГРУЗКА МОДЕЛИ ДЛЯ САЙТА =====
    print("\n[3] Загрузка модели для веб-приложения")
    model = load_local_model()

    # ===== ОЦЕНКА =====
    print("\n[4] Финальная оценка модели")
    metrics = evaluate_model(model)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n ML SYSTEM END ")
    print("=" * 50)


if __name__ == "__main__":
    main()
