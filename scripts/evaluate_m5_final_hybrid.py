
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "data/processed/m5/m5_train.csv"
TEST_FILE = BASE_DIR / "data/processed/m5/m5_test.csv"

MODEL_FILE = (
    BASE_DIR
    / "models/demand_forecasting/m5_hist_gradient_boosting.joblib"
)

OUTPUT_FILE = (
    BASE_DIR
    / "models/demand_forecasting/m5_final_comparison.json"
)

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

TARGET = "units_sold"
CHUNK_SIZE = 100_000


def calculate_tiers():
    print("Calculating demand tiers from training data...")

    sales_by_product = defaultdict(float)

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=["id", TARGET],
        chunksize=CHUNK_SIZE,
    ):
        grouped = chunk.groupby("id", observed=True)[TARGET].sum()

        for product_id, sales in grouped.items():
            sales_by_product[product_id] += sales

    training = pd.DataFrame({
        "id": list(sales_by_product.keys()),
        "training_sales": list(sales_by_product.values()),
    })

    training["demand_tier"] = pd.qcut(
        training["training_sales"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
    )

    return dict(zip(training["id"], training["demand_tier"]))


def main():
    print("Loading final ML model...")
    model = joblib.load(MODEL_FILE)

    tiers = calculate_tiers()

    metrics = {
        name: {
            "absolute_error": 0.0,
            "squared_error": 0.0,
            "actual_total": 0.0,
        }
        for name in ["ML", "Baseline", "Hybrid"]
    }

    total_rows = 0
    print("\nEvaluating untouched final test dataset...")

    for chunk in pd.read_csv(
        TEST_FILE,
        usecols=["id"] + FEATURES + [TARGET],
        chunksize=CHUNK_SIZE,
    ):
        X = chunk[FEATURES].astype("float32")

        actual = chunk[TARGET].to_numpy(dtype="float64")

        ml_predictions = model.predict(X)

        baseline_predictions = chunk[
            "rolling_mean_7"
        ].to_numpy(dtype="float64")

        # Fixed rule selected using validation results:
        # Low and Medium -> Rolling Mean 7
        # High and Very High -> ML model

        product_tiers = chunk["id"].map(tiers)

        if product_tiers.isna().any():
            raise ValueError("Some test products have no training demand tier.")

        use_baseline = product_tiers.isin(["Low", "Medium"]).to_numpy()

        hybrid_predictions = np.where(
            use_baseline,
            baseline_predictions,
            ml_predictions,
        )

        predictions = {
            "ML": ml_predictions,
            "Baseline": baseline_predictions,
            "Hybrid": hybrid_predictions,
        }

        for name, predicted in predictions.items():
            errors = actual - predicted

            metrics[name]["absolute_error"] += np.abs(errors).sum()
            metrics[name]["squared_error"] += np.square(errors).sum()
            metrics[name]["actual_total"] += np.abs(actual).sum()

        total_rows += len(chunk)

    results = {
        "test_rows": total_rows,
        "products": len(tiers),
        "test_period": "2015-04-14 to 2016-04-24",
        "hybrid_rule": {
            "Low": "Rolling Mean 7",
            "Medium": "Rolling Mean 7",
            "High": "HistGradientBoosting",
            "Very High": "HistGradientBoosting",
        },
    }

    print("\nFINAL TEST RESULTS")
    print("=" * 60)

    for name, values in metrics.items():
        mae = values["absolute_error"] / total_rows

        rmse = np.sqrt(
            values["squared_error"] / total_rows
        )

        wape = (
            values["absolute_error"]
            / values["actual_total"] * 100
        )

        results[name] = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "WAPE": round(float(wape), 4),
        }

        print(f"\n{name}")
        print("MAE:", round(float(mae), 4))
        print("RMSE:", round(float(rmse), 4))
        print("WAPE:", round(float(wape), 4))

    with open(OUTPUT_FILE, "w") as file:
        import json
        json.dump(results, file, indent=4)

    print("\nFinal results saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
