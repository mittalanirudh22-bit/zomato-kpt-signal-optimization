from src.data_generation import generate_synthetic_data
from src.signal_engineering import feature_engineering
from src.kitchen_load import compute_kli
from src.model import train_model
from src.evaluation import evaluate_baseline
import matplotlib.pyplot as plt

def main():

    print("Generating dataset...")
    df = generate_synthetic_data()
    df.to_csv("data/raw/synthetic_orders.csv", index=False)

    print("Engineering features...")
    df = feature_engineering(df)

    print("Computing Kitchen Load Index...")
    df = compute_kli(df)
    df.to_csv("data/processed/processed_orders.csv", index=False)

    print("Training ML model...")
    model, improved_mae = train_model(df)

    print("Evaluating baseline...")
    baseline_mae = evaluate_baseline(df)

    print("\n--- RESULTS ---")
    print("Baseline MAE:", baseline_mae)
    print("Improved MAE:", improved_mae)
    with open("results/metrics.txt", "w") as f:
     f.write(f"Baseline MAE: {baseline_mae}\n")
     f.write(f"Improved MAE: {improved_mae}\n")

    plt.bar(["Baseline", "Improved"], [baseline_mae, improved_mae])
    plt.title("MAE Comparison")
    plt.ylabel("Mean Absolute Error")
    plt.savefig("results/mae_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()
