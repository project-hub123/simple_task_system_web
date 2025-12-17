from ml.train import train_model_v1, train_model_v2
from ml_model import load_local_model, evaluate_model

def main():
    print("=== ML SYSTEM START ===")

    print("\n[1] Обучение модели V1 (train/test)")
    model_v1, acc_v1 = train_model_v1()
    print(f"V1 accuracy: {acc_v1:.4f}")

    print("\n[2] Обучение модели V2 (полный датасет)")
    model_v2, acc_v2 = train_model_v2()
    print(f"V2 accuracy: {acc_v2:.4f}")

    print("\n[3] Загрузка модели для сайта")
    model = load_local_model()

    print("\n[4] Оценка модели")
    metrics = evaluate_model(model)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== ML SYSTEM END ===")

if __name__ == "__main__":
    main()
