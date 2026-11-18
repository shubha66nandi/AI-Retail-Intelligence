
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "data/processed/m5/m5_train.csv"
TEST_FILE = BASE_DIR / "data/processed/m5/m5_test.csv"

MODEL_DIR = BASE_DIR / "models/demand_forecasting"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "m5_hist_gradient_boosting.joblib"
RESULT_FILE = MODEL_DIR / "m5_model_results.json"

CHUNK_SIZE = 100_000
SAMPLE_FRACTION = 0.23

TARGET = "units_sold"

FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
    "sell_price",
    "price_missing",
]

COLUMNS = FEATURES + [TARGET]


def load_training_sample():
    print("Loading training data in chunks...")

    samples = []

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=COLUMNS,
        chunksize=CHUNK_SIZE,
    ):
        sample = chunk.sample(
            frac=SAMPLE_FRACTION,
            random_state=42,
        )

        samples.append(sample)

    train = pd.concat(samples, ignore_index=True)

    print("Training sample rows:", len(train))

    X_train = train[FEATURES].astype("float32")
    y_train = train[TARGET].astype("float32")

    return X_train, y_train


def train_model(X_train, y_train):
    print("\nInitializing HistGradientBoostingRegressor...")

    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.08,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    print("Training model...")

    model.fit(X_train, y_train)

    print("Model training completed.")

    return model


def evaluate_model(model):
    print("\nEvaluating on complete test dataset...")

    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_actual = 0.0
    total_rows = 0

    for chunk in pd.read_csv(
        TEST_FILE,
        usecols=COLUMNS,
        chunksize=CHUNK_SIZE,
    ):
        X_test = chunk[FEATURES].astype("float32")
        y_test = chunk[TARGET].to_numpy(dtype="float64")

        predictions = model.predict(X_test)

        errors = y_test - predictions

        total_absolute_error += np.abs(errors).sum()
        total_squared_error += np.square(errors).sum()
        total_actual += np.abs(y_test).sum()
        total_rows += len(chunk)

    mae = total_absolute_error / total_rows
    rmse = np.sqrt(total_squared_error / total_rows)
    wape = (
        total_absolute_error / total_actual * 100
        if total_actual > 0 else float("nan")
    )

    return {
        "model": "HistGradientBoostingRegressor",
        "training_sample_rows": int(total_rows * 0 + 0),
        "test_rows": total_rows,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "WAPE": round(float(wape), 4),
    }


def main():
    X_train, y_train = load_training_sample()

    model = train_model(X_train, y_train)

    print("\nSaving trained model...")

    joblib.dump(model, MODEL_FILE)

    print("Model saved:", MODEL_FILE)

    results = evaluate_model(model)
    results["training_sample_rows"] = len(X_train)

    with open(RESULT_FILE, "w") as file:
        json.dump(results, file, indent=4)

    print("\nMODEL EVALUATION RESULTS")
    print("=" * 40)

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nResults saved:", RESULT_FILE)


if __name__ == "__main__":
    main()
