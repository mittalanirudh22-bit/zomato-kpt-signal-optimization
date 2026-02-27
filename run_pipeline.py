from src.data_generation import generate_synthetic_data
from src.signal_engineering import feature_engineering
from src.kitchen_load import compute_kli
from src.model import train_model
from src.evaluation import evaluate_baseline
import matplotlib.pyplot as plt
import numpy as np

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

    # --- Simulate Operational Impact Metrics ---

    # Simulate rider arrival (assume rider dispatched 10 mins after order)
    rider_arrival_time = 10  

    # Baseline prediction (naive rule)
    baseline_pred = df["items_count"] * 10

    # Improved prediction (approximate using model)
    features = [
    "complexity_score",
    "kitchen_load_index",
    "merchant_reliability",
    "external_load"
  ] 
    improved_pred = model.predict(df[features])

    # Rider waiting time (if rider arrives before food ready)
    baseline_wait = np.maximum(baseline_pred - rider_arrival_time, 0)
    improved_wait = np.maximum(improved_pred - rider_arrival_time, 0)

    # Cancellation probability proxy (long waits increase cancellation risk)
    baseline_cancel_rate = (baseline_wait > 15).mean()
    improved_cancel_rate = (improved_wait > 15).mean()

    # Rider idle time proxy (early arrival before food ready)
    baseline_idle = np.maximum(rider_arrival_time - baseline_pred, 0)
    improved_idle = np.maximum(rider_arrival_time - improved_pred, 0)

    print("\n--- Operational Impact ---")
    print("Baseline Avg Wait Time:", baseline_wait.mean())
    print("Improved Avg Wait Time:", improved_wait.mean())
    print("Baseline Cancel Rate:", baseline_cancel_rate)
    print("Improved Cancel Rate:", improved_cancel_rate)
    print("Baseline Idle Time:", baseline_idle.mean())
    print("Improved Idle Time:", improved_idle.mean())

    # Save operational metrics
    with open("results/operational_metrics.txt", "w") as f:
     f.write(f"Baseline Avg Wait: {baseline_wait.mean()}\n")
     f.write(f"Improved Avg Wait: {improved_wait.mean()}\n")
     f.write(f"Baseline Cancel Rate: {baseline_cancel_rate}\n")
     f.write(f"Improved Cancel Rate: {improved_cancel_rate}\n")
     f.write(f"Baseline Idle Time: {baseline_idle.mean()}\n")
     f.write(f"Improved Idle Time: {improved_idle.mean()}\n") 

    plt.bar(["Baseline", "Improved"], [baseline_mae, improved_mae])
    plt.title("MAE Comparison")
    plt.ylabel("Mean Absolute Error")
    plt.savefig("results/mae_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()
