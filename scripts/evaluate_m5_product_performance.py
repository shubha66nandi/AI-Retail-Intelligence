
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TEST_FILE = BASE_DIR / "data/processed/m5/m5_test.csv"
MODEL_FILE = BASE_DIR / "models/demand_forecasting/m5_hist_gradient_boosting.joblib"

OUTPUT_DIR = BASE_DIR / "models/demand_forecasting"
OUTPUT_FILE = OUTPUT_DIR / "m5_model_vs_baseline.csv"

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


def create_metrics():
    return {
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "actual_total": 0.0,
        "rows": 0,
    }


def main():
    print("Loading trained model...")
    model = joblib.load(MODEL_FILE)

    product_metrics = defaultdict(
        lambda: {
            "model": create_metrics(),
            "baseline": create_metrics(),
        }
    )

    print("Comparing ML model with Rolling Mean 7...")

    for chunk in pd.read_csv(
        TEST_FILE,
        usecols=["id"] + FEATURES + [TARGET],
        chunksize=CHUNK_SIZE,
    ):

        X = chunk[FEATURES].astype("float32")

        actual = chunk[TARGET].to_numpy(dtype="float64")

        model_predictions = model.predict(X)

        baseline_predictions = chunk[
            "rolling_mean_7"
        ].to_numpy(dtype="float64")

        model_errors = actual - model_predictions
        baseline_errors = actual - baseline_predictions

        chunk_results = pd.DataFrame({
            "id": chunk["id"].to_numpy(),
            "actual": np.abs(actual),
            "model_abs_error": np.abs(model_errors),
            "model_squared_error": np.square(model_errors),
            "baseline_abs_error": np.abs(baseline_errors),
            "baseline_squared_error": np.square(baseline_errors),
        })

        grouped = chunk_results.groupby("id").agg(
            actual_total=("actual", "sum"),
            model_abs_error=("model_abs_error", "sum"),
            model_squared_error=("model_squared_error", "sum"),
            baseline_abs_error=("baseline_abs_error", "sum"),
            baseline_squared_error=("baseline_squared_error", "sum"),
            rows=("actual", "size"),
        )

        for product_id, row in grouped.iterrows():

            result = product_metrics[product_id]

            for method, prefix in [
                ("model", "model"),
                ("baseline", "baseline"),
            ]:
                result[method]["absolute_error"] += row[
                    f"{prefix}_abs_error"
                ]

                result[method]["squared_error"] += row[
                    f"{prefix}_squared_error"
                ]

                result[method]["actual_total"] += row["actual_total"]
                result[method]["rows"] += int(row["rows"])

    print("Calculating product-level metrics...")

    results = []

    for product_id, values in product_metrics.items():

        model_stats = values["model"]
        baseline_stats = values["baseline"]

        rows = model_stats["rows"]
        actual_total = model_stats["actual_total"]

        model_mae = model_stats["absolute_error"] / rows
        model_rmse = np.sqrt(model_stats["squared_error"] / rows)

        baseline_mae = baseline_stats["absolute_error"] / rows
        baseline_rmse = np.sqrt(baseline_stats["squared_error"] / rows)

        model_wape = (
            model_stats["absolute_error"] / actual_total * 100
            if actual_total > 0 else np.nan
        )

        baseline_wape = (
            baseline_stats["absolute_error"] / actual_total * 100
            if actual_total > 0 else np.nan
        )

        results.append({
            "product_id": product_id,
            "rows": rows,
            "total_actual_sales": round(actual_total, 2),
            "ML_MAE": round(model_mae, 4),
            "Baseline_MAE": round(baseline_mae, 4),
            "ML_RMSE": round(model_rmse, 4),
            "Baseline_RMSE": round(baseline_rmse, 4),
            "ML_WAPE": round(model_wape, 4) if not np.isnan(model_wape) else np.nan,
            "Baseline_WAPE": round(baseline_wape, 4) if not np.isnan(baseline_wape) else np.nan,
            "ML_better_MAE": model_mae < baseline_mae,
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nMODEL VS BASELINE SUMMARY")
    print("=" * 55)

    print("Products evaluated:", len(results_df))

    print(
        "Products where ML has lower MAE:",
        int(results_df["ML_better_MAE"].sum())
    )

    print(
        "Products where baseline has lower or equal MAE:",
        int((~results_df["ML_better_MAE"]).sum())
    )

    print("\nAVERAGE PRODUCT MAE")
    print("ML Model:", round(results_df["ML_MAE"].mean(), 4))
    print("Baseline:", round(results_df["Baseline_MAE"].mean(), 4))

    print("\nAVERAGE PRODUCT WAPE")
    print("ML Model:", round(results_df["ML_WAPE"].mean(), 4))
    print("Baseline:", round(results_df["Baseline_WAPE"].mean(), 4))

    print("\nBEST 10 PRODUCTS WHERE ML IMPROVES MAE")

    improved = results_df[
        results_df["ML_better_MAE"]
    ].sort_values("ML_MAE")

    print(improved.head(10).to_string(index=False))

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
