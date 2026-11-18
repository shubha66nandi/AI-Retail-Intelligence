from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

BASE = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE / "data/processed/m5/m5_v2_model_train.csv"
VAL_FILE = BASE / "data/processed/m5/m5_v2_validation.csv"

MODEL_DIR = BASE / "models/demand_forecasting"

MODEL_FILE = MODEL_DIR / "m5_v2_validation_model.joblib"
RESULT_FILE = MODEL_DIR / "m5_v2_validation_results.json"

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
    "sell_price",
    "price_missing",
    "price_change_pct",
    "snap_CA",
    "event_national",
    "event_religious",
    "event_cultural",
    "event_sporting",
]

TARGET = "units_sold"
CHUNK_SIZE = 100_000
SAMPLE_FRACTION = 0.23


def load_training_sample():
    print("Loading V2 training data...")

    samples = []

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=FEATURES + [TARGET],
        chunksize=CHUNK_SIZE,
    ):
        samples.append(
            chunk.sample(
                frac=SAMPLE_FRACTION,
                random_state=42,
            )
        )

    train = pd.concat(samples, ignore_index=True)

    X_train = train[FEATURES].astype("float32")
    y_train = train[TARGET].astype("float32")

    print("Training sample rows:", len(train))

    return X_train, y_train


def main():
    X_train, y_train = load_training_sample()

    print("\nTraining V2 HistGradientBoosting model...")

    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.08,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    print("Model saved:", MODEL_FILE)

    print("\nEvaluating on V2 validation dataset...")

    total_error = 0.0
    total_squared_error = 0.0
    total_baseline_error = 0.0
    total_baseline_squared_error = 0.0
    total_actual = 0.0
    total_rows = 0

    for chunk in pd.read_csv(
        VAL_FILE,
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
        "model": "HistGradientBoostingRegressor_V2",
        "features": FEATURES,
        "training_sample_rows": len(X_train),
        "validation_rows": total_rows,
        "ML_MAE": total_error / total_rows,
        "ML_RMSE": np.sqrt(
            total_squared_error / total_rows
        ),
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

    print("\nV2 VALIDATION RESULTS")
    print("=" * 50)

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nResults saved:", RESULT_FILE)


if __name__ == "__main__":
    main()
