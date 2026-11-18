
from pathlib import Path
import numpy as np
import pandas as pd
import json

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data/processed/m5/m5_test.csv"
OUTPUT_DIR = BASE_DIR / "models/demand_forecasting"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "units_sold"

BASELINES = {
    "Lag 1": "lag_1",
    "Lag 7": "lag_7",
    "Lag 28": "lag_28",
    "Rolling Mean 7": "rolling_mean_7",
    "Rolling Mean 28": "rolling_mean_28",
}


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    total_actual = np.sum(np.abs(actual))
    wape = (
        np.sum(np.abs(actual - predicted)) / total_actual * 100
        if total_actual > 0 else np.nan
    )

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "WAPE": round(float(wape), 4),
    }


def main():
    print("Loading test dataset...")

    df = pd.read_csv(DATA_FILE)

    print("Test rows:", len(df))

    results = []

    for name, column in BASELINES.items():
        print(f"Evaluating {name}...")

        metrics = calculate_metrics(
            df[TARGET],
            df[column]
        )

        results.append({
            "Model": name,
            **metrics
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE")

    print("\nM5 BASELINE RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    output_file = OUTPUT_DIR / "m5_baseline_results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()
