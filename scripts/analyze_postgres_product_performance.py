
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT_DIR / "data" / "processed" / "postgres_predictions.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "postgres_product_performance.csv"


def calculate_metrics(group, prediction_column):
    actual = group["units_sold"]
    predicted = group[prediction_column]

    absolute_errors = (actual - predicted).abs()

    total_actual = actual.sum()

    wape = (
        absolute_errors.sum() / total_actual * 100
        if total_actual > 0
        else np.nan
    )

    return {
        "MAE": absolute_errors.mean(),
        "RMSE": np.sqrt(((actual - predicted) ** 2).mean()),
        "WAPE": wape,
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "product_id",
        "sku",
        "product_name",
        "units_sold",
        "ml_prediction",
        "baseline_prediction",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results = []

    for _, group in df.groupby(
        ["product_id", "sku", "product_name"]
    ):
        ml = calculate_metrics(group, "ml_prediction")
        baseline = calculate_metrics(group, "baseline_prediction")

        results.append({
            "product_id": group["product_id"].iloc[0],
            "sku": group["sku"].iloc[0],
            "product_name": group["product_name"].iloc[0],
            "test_days": len(group),
            "actual_units": group["units_sold"].sum(),
            "ml_mae": ml["MAE"],
            "ml_rmse": ml["RMSE"],
            "ml_wape": ml["WAPE"],
            "baseline_mae": baseline["MAE"],
            "baseline_rmse": baseline["RMSE"],
            "baseline_wape": baseline["WAPE"],
            "ml_better": ml["WAPE"] < baseline["WAPE"],
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("ml_wape")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nPRODUCT-WISE FORECASTING PERFORMANCE")
    print("=" * 85)

    print(
        result_df[
            [
                "sku",
                "product_name",
                "ml_wape",
                "baseline_wape",
                "ml_better",
            ]
        ].to_string(index=False)
    )

    print("\nSUMMARY")
    print("=" * 85)

    print(f"Total products evaluated: {len(result_df)}")
    print(f"ML better: {result_df['ml_better'].sum()}")
    print(f"Baseline better: {(~result_df['ml_better']).sum()}")

    best = result_df.iloc[0]
    worst = result_df.iloc[-1]

    print(
        f"\nBest ML WAPE: {best['product_name']} "
        f"({best['ml_wape']:.2f}%)"
    )

    print(
        f"Worst ML WAPE: {worst['product_name']} "
        f"({worst['ml_wape']:.2f}%)"
    )

    print(f"\nSaved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
