
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT_DIR / "data" / "processed" / "postgres_daily_sales.csv"
MODEL_FILE = ROOT_DIR / "models" / "demand_forecasting" / "postgres_hgb_model.joblib"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "next_day_forecast.csv"

FORECAST_DATE = pd.Timestamp("2026-06-30")

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


def create_latest_features(group):
    group = group.sort_values("date").reset_index(drop=True)

    if len(group) < 28:
        raise ValueError(
            f"Insufficient history for product: {group['sku'].iloc[0]}"
        )

    sales = group["units_sold"]

    latest = {
        "lag_1": sales.iloc[-1],
        "lag_7": sales.iloc[-7],
        "lag_14": sales.iloc[-14],
        "lag_28": sales.iloc[-28],
        "rolling_mean_7": sales.tail(7).mean(),
        "rolling_mean_14": sales.tail(14).mean(),
        "rolling_mean_28": sales.tail(28).mean(),
        "rolling_std_7": sales.tail(7).std(),
        "rolling_std_28": sales.tail(28).std(),
        "day_of_week": FORECAST_DATE.dayofweek,
        "month": FORECAST_DATE.month,
    }

    return latest


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Daily sales file not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])

    required = {
        "product_id",
        "sku",
        "product_name",
        "date",
        "units_sold",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    model = joblib.load(MODEL_FILE)

    forecast_rows = []

    for _, group in df.groupby(
        ["product_id", "sku", "product_name"]
    ):
        latest_features = create_latest_features(group)

        feature_df = pd.DataFrame(
            [latest_features],
            columns=FEATURES,
        )

        prediction = model.predict(feature_df)[0]
        prediction = max(0, prediction)

        forecast_rows.append({
            "product_id": group["product_id"].iloc[0],
            "sku": group["sku"].iloc[0],
            "product_name": group["product_name"].iloc[0],
            "forecast_date": FORECAST_DATE.date(),
            "predicted_units": round(float(prediction), 2),
        })

    result = pd.DataFrame(forecast_rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nNEXT-DAY DEMAND FORECAST")
    print("=" * 75)
    print(result.to_string(index=False))

    print("\nSUMMARY")
    print("=" * 75)
    print(f"Forecast date: {FORECAST_DATE.date()}")
    print(f"Products forecasted: {len(result)}")
    print(f"Total predicted demand: {result['predicted_units'].sum():.2f} units")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
