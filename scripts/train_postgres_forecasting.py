
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE / "data/processed/postgres_train.csv"
TEST_FILE = BASE / "data/processed/postgres_test.csv"

MODEL_DIR = BASE / "models/demand_forecasting"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "postgres_hgb_model.joblib"
RESULT_FILE = MODEL_DIR / "postgres_forecasting_results.json"

PREDICTION_FILE = BASE / "data/processed/postgres_predictions.csv"


FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "day_of_week",
    "month",
]

TARGET = "units_sold"


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    total_actual = np.abs(actual).sum()

    wape = (
        np.abs(actual - predicted).sum()
        / total_actual * 100
        if total_actual > 0 else 0
    )

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "WAPE": round(float(wape), 4),
    }


def main():

    print("Loading training and testing datasets...")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    X_train = train[FEATURES].astype("float32")
    y_train = train[TARGET].astype("float32")

    X_test = test[FEATURES].astype("float32")
    y_test = test[TARGET].to_numpy(dtype="float64")

    print("Training rows:", len(train))
    print("Testing rows:", len(test))

    # Train model
    print("\nTraining HistGradientBoosting model...")

    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)

    print("Model saved:", MODEL_FILE)

    # Predictions
    print("\nGenerating predictions...")

    ml_predictions = np.maximum(
        model.predict(X_test),
        0
    )

    baseline_predictions = test[
        "rolling_mean_7"
    ].to_numpy(dtype="float64")

    baseline_predictions = np.maximum(
        baseline_predictions,
        0
    )

    # Evaluation
    ml_metrics = calculate_metrics(
        y_test,
        ml_predictions
    )

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions
    )

    results = {
        "model": "HistGradientBoostingRegressor",
        "features": FEATURES,
        "training_rows": len(train),
        "testing_rows": len(test),
        "test_start": str(test["date"].min()),
        "test_end": str(test["date"].max()),
        "ML_Model": ml_metrics,
        "Rolling_Mean_7_Baseline": baseline_metrics,
    }

    with open(RESULT_FILE, "w") as file:
        json.dump(results, file, indent=4)

    # Save predictions
    output = test[
        ["product_id", "sku", "product_name", "date", TARGET]
    ].copy()

    output["ml_prediction"] = ml_predictions
    output["baseline_prediction"] = baseline_predictions

    output.to_csv(PREDICTION_FILE, index=False)

    print("\nFORECASTING EVALUATION RESULTS")
    print("=" * 50)

    print("\nHistGradientBoosting:")
    for key, value in ml_metrics.items():
        print(f"{key}: {value}")

    print("\nRolling Mean 7 Baseline:")
    for key, value in baseline_metrics.items():
        print(f"{key}: {value}")

    print("\nResults saved:", RESULT_FILE)
    print("Predictions saved:", PREDICTION_FILE)


if __name__ == "__main__":
    main()
