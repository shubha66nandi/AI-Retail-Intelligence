
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

BASE = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE / "data/processed/m5/m5_v2_train.csv"
TEST_FILE = BASE / "data/processed/m5/m5_v2_test.csv"

MODEL_DIR = BASE / "models/demand_forecasting"

MODEL_FILE = MODEL_DIR / "m5_v2_final_model.joblib"
RESULT_FILE = MODEL_DIR / "m5_v2_final_results.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "lag_56",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_std_7",
    "rolling_std_28",
    "day_of_week",
    "month",
    "price_change_pct",
    "snap_CA",
    "event_national",
    "event_religious",
    "event_cultural",
    "event_sporting",
]

TARGET = "units_sold"
CHUNK_SIZE = 100_000


def load_training_data():
    print("Loading V2 training sample...")

    samples = []

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=FEATURES + [TARGET],
        chunksize=CHUNK_SIZE,
    ):
        samples.append(
            chunk.sample(
                frac=0.23,
                random_state=42,
            )
        )

    train = pd.concat(samples, ignore_index=True)

    X_train = train[FEATURES].astype("float32")
    y_train = train[TARGET].astype("float32")

    print("Training sample rows:", len(train))

    return X_train, y_train


def main():
    print("Loading previously trained V2 model...")

    model = joblib.load(MODEL_FILE)

    training_rows = 489155

    print("Model loaded:", MODEL_FILE)
    print("Training sample rows:", training_rows)
    print("\nEvaluating on untouched V2 final-test dataset...")

    total_error = 0.0
    total_squared_error = 0.0
    total_baseline_error = 0.0
    total_baseline_squared_error = 0.0
    total_actual = 0.0
    total_rows = 0

    for chunk in pd.read_csv(
        TEST_FILE,
        usecols=FEATURES + [TARGET],
        chunksize=CHUNK_SIZE,
    ):
        X = chunk[FEATURES].astype("float32")
        actual = chunk[TARGET].to_numpy(dtype="float64")

        predictions = model.predict(X)

        baseline = chunk["rolling_mean_7"].to_numpy(
            dtype="float64"
        )

        errors = actual - predictions
        baseline_errors = actual - baseline

        total_error += np.abs(errors).sum()
        total_squared_error += np.square(errors).sum()

        total_baseline_error += np.abs(baseline_errors).sum()
        total_baseline_squared_error += np.square(
            baseline_errors
        ).sum()

        total_actual += np.abs(actual).sum()
        total_rows += len(chunk)

    results = {
        "model": "HistGradientBoostingRegressor_V2_Final",
        "features": FEATURES,
        "training_rows": training_rows,
        "test_rows": total_rows,
        "test_start": "2015-04-14",
        "test_end": "2016-04-24",
        "ML_MAE": total_error / total_rows,
        "ML_RMSE": np.sqrt(total_squared_error / total_rows),
        "ML_WAPE": total_error / total_actual * 100,
        "Baseline_MAE": total_baseline_error / total_rows,
        "Baseline_RMSE": np.sqrt(
            total_baseline_squared_error / total_rows
        ),
        "Baseline_WAPE": (
            total_baseline_error / total_actual * 100
        ),
    }

    results = {
        key: round(float(value), 4)
        if isinstance(value, (float, np.floating))
        else value
        for key, value in results.items()
    }

    with open(RESULT_FILE, "w") as file:
        json.dump(results, file, indent=4)

    print("\nV2 FINAL TEST RESULTS")
    print("=" * 50)

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nResults saved:", RESULT_FILE)


if __name__ == "__main__":
    main()
