from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def train_model(df):

    features = [
        "complexity_score",
        "kitchen_load_index",
        "merchant_reliability",
        "external_load"
    ]

    X = df[features]
    y = df["actual_kpt"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = XGBRegressor(n_estimators=200, max_depth=6)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    return model, mae