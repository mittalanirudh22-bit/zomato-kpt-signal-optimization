from sklearn.metrics import mean_absolute_error

def evaluate_baseline(df):

    baseline_pred = df["items_count"] * 10
    mae = mean_absolute_error(df["actual_kpt"], baseline_pred)

    return mae