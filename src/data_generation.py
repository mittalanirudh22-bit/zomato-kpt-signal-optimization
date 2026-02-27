import pandas as pd
import numpy as np

def generate_synthetic_data(n=5000, seed=42):
    np.random.seed(seed)

    df = pd.DataFrame({
        "restaurant_id": np.random.randint(1, 200, n),
        "items_count": np.random.randint(1, 8, n),
        "prep_complexity": np.random.randint(1, 5, n),
        "time_of_day": np.random.randint(0, 24, n),
        "orders_last_10min": np.random.randint(0, 20, n),
        "external_load": np.random.randint(0, 15, n),
        "merchant_reliability": np.random.uniform(0.6, 1.0, n),
        "rider_distance_km": np.random.uniform(0.5, 5, n)
    })

    df["actual_kpt"] = (
        0.8 * df["items_count"] * df["prep_complexity"] +
        0.6 * df["orders_last_10min"] +
        0.5 * df["external_load"] -
        2 * df["merchant_reliability"] +
        np.random.normal(0, 2, n)
    )

    return df

