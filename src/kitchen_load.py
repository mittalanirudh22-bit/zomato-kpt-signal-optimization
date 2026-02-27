def compute_kli(df):

    df["kitchen_load_index"] = (
        0.4 * df["orders_last_10min"] +
        0.3 * df["complexity_score"] +
        0.2 * df["external_load"] +
        0.1 * df["merchant_speed_inverse"]
    ) * df["rush_factor"]

    return df