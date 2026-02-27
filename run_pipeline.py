from src.data_generation import generate_synthetic_data
from src.signal_engineering import feature_engineering
from src.kitchen_load import compute_kli
from src.model import train_model
from src.evaluation import evaluate_baseline

def main():

    print("Generating dataset...")
    df = generate_synthetic_data()

    print("Engineering features...")
    df = feature_engineering(df)

    print("Computing Kitchen Load Index...")
    df = compute_kli(df)

    print("Training ML model...")
    model, improved_mae = train_model(df)

    print("Evaluating baseline...")
    baseline_mae = evaluate_baseline(df)

    print("\n--- RESULTS ---")
    print("Baseline MAE:", baseline_mae)
    print("Improved MAE:", improved_mae)

if __name__ == "__main__":
    main()