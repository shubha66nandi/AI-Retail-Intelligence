
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "data/processed/m5/m5_model_train.csv"
VALIDATION_FILE = BASE_DIR / "data/processed/m5/m5_validation.csv"

MODEL_FILE = BASE_DIR / "models/demand_forecasting/m5_validation_model.joblib"

OUTPUT_FILE = (
    BASE_DIR
    / "models/demand_forecasting/m5_validation_tier_comparison.csv"
)

FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "day_of_week", "month", "sell_price", "price_missing",
]

TARGET = "units_sold"
CHUNK_SIZE = 100_000


def main():
    print("Loading validation model...")
    model = joblib.load(MODEL_FILE)

    print("Calculating training-period demand by product...")

    training_sales = defaultdict(float)

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=["id", TARGET],
        chunksize=CHUNK_SIZE,
    ):
        grouped = chunk.groupby("id", observed=True)[TARGET].sum()

        for product_id, sales in grouped.items():
            training_sales[product_id] += sales

    training = pd.DataFrame({
        "id": list(training_sales.keys()),
        "training_sales": list(training_sales.values()),
    })

    training["demand_tier"] = pd.qcut(
        training["training_sales"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
    )

    print("Evaluating validation products...")

    metrics = defaultdict(lambda: {
        "ml_error": 0.0,
        "ml_squared_error": 0.0,
        "baseline_error": 0.0,
        "baseline_squared_error": 0.0,
        "actual_total": 0.0,
        "rows": 0,
    })

    for chunk in pd.read_csv(
        VALIDATION_FILE,
        usecols=["id"] + FEATURES + [TARGET],
        chunksize=CHUNK_SIZE,
    ):
        X = chunk[FEATURES].astype("float32")

        actual = chunk[TARGET].to_numpy(dtype="float64")

        ml_predictions = model.predict(X)

        baseline_predictions = chunk[
            "rolling_mean_7"
        ].to_numpy(dtype="float64")

        ml_errors = actual - ml_predictions
        baseline_errors = actual - baseline_predictions

        temp = pd.DataFrame({
            "id": chunk["id"].to_numpy(),
            "actual": np.abs(actual),
            "ml_error": np.abs(ml_errors),
            "ml_squared": np.square(ml_errors),
            "baseline_error": np.abs(baseline_errors),
            "baseline_squared": np.square(baseline_errors),
        })

        grouped = temp.groupby("id").agg(
            actual=("actual", "sum"),
            ml_error=("ml_error", "sum"),
            ml_squared=("ml_squared", "sum"),
            baseline_error=("baseline_error", "sum"),
            baseline_squared=("baseline_squared", "sum"),
            rows=("actual", "size"),
        )

        for product_id, row in grouped.iterrows():
            result = metrics[product_id]

            result["ml_error"] += row["ml_error"]
            result["ml_squared_error"] += row["ml_squared"]
            result["baseline_error"] += row["baseline_error"]
            result["baseline_squared_error"] += row["baseline_squared"]
            result["actual_total"] += row["actual"]
            result["rows"] += int(row["rows"])

    results = []

    for product_id, values in metrics.items():
        rows = values["rows"]
        actual_total = values["actual_total"]

        results.append({
            "id": product_id,
            "actual_sales": actual_total,
            "ML_MAE": values["ml_error"] / rows,
            "Baseline_MAE": values["baseline_error"] / rows,
            "ML_error": values["ml_error"],
            "Baseline_error": values["baseline_error"],
            "ML_WAPE": (
                values["ml_error"] / actual_total * 100
                if actual_total else np.nan
            ),
            "Baseline_WAPE": (
                values["baseline_error"] / actual_total * 100
                if actual_total else np.nan
            ),
        })

    results = pd.DataFrame(results)

    results = results.merge(
        training,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    print("\nVALIDATION PERFORMANCE BY TRAINING DEMAND TIER")
    print("=" * 70)

    summary = results.groupby(
        "demand_tier", observed=True
    ).apply(
        lambda g: pd.Series({
            "Products": len(g),
            "ML_MAE": g["ML_MAE"].mean(),
            "Baseline_MAE": g["Baseline_MAE"].mean(),
            "ML_WAPE": g["ML_error"].sum() / g["actual_sales"].sum() * 100
            if g["actual_sales"].sum() else np.nan,
            "Baseline_WAPE": g["Baseline_error"].sum() / g["actual_sales"].sum() * 100
            if g["actual_sales"].sum() else np.nan,
            "ML_better_MAE": (g["ML_MAE"] < g["Baseline_MAE"]).sum(),
        }),
        include_groups=False,
    )

    print(summary.round(4).to_string())

    # Test a hybrid rule: baseline for Low, ML for all other tiers.
    results["Hybrid_error"] = np.where(
        results["demand_tier"].isin(["Low", "Medium"]),
        results["Baseline_error"],
        results["ML_error"],
    )

    total_actual = results["actual_sales"].sum()

    hybrid_wape = (
        results["Hybrid_error"].sum() / total_actual * 100
    )

    print("\nOVERALL VALIDATION WAPE")
    print("ML:", round(results["ML_error"].sum() / total_actual * 100, 4))
    print("Baseline:", round(results["Baseline_error"].sum() / total_actual * 100, 4))
    print("Hybrid:", round(hybrid_wape, 4))

    results.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
