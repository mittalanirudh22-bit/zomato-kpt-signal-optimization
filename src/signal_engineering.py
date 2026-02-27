def feature_engineering(df):

    df["complexity_score"] = df["items_count"] * df["prep_complexity"]

    df["rush_factor"] = df["time_of_day"].apply(
        lambda x: 1.5 if x in [12,13,14,19,20,21] else 1
    )

    df["merchant_speed_inverse"] = 1 / df["merchant_reliability"]

    return df
