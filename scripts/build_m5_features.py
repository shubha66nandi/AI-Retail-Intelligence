
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "m5"
    / "ca1_foods_daily.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "m5"
OUTPUT_FILE = OUTPUT_DIR / "m5_features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading M5 dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "id",
            "date",
            "units_sold",
            "sell_price",
            "weekday",
        ],
        parse_dates=["date"],
        dtype={
            "id": "category",
            "units_sold": "int16",
            "sell_price": "float32",
            "weekday": "category",
        },
    )

    print("Sorting product sales by date...")

    df = df.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    grouped = df.groupby("id", observed=True)["units_sold"]

    print("Creating lag features...")

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = grouped.shift(lag)

    print("Creating rolling mean features...")

    df["rolling_mean_7"] = df.groupby(
        "id", observed=True
    )["units_sold"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df["rolling_mean_28"] = df.groupby(
        "id", observed=True
    )["units_sold"].transform(
        lambda x: x.shift(1).rolling(28).mean()
    )

    print("Creating calendar features...")

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Keep missing prices as NaN.
    # Missing price does not mean zero price.
    df["price_missing"] = df["sell_price"].isna().astype("int8")

    print("Removing rows without 28-day history...")

    df = df.dropna(
        subset=[
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_28",
        ]
    ).copy()

    print("Saving feature dataset...")

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nM5 FEATURE ENGINEERING COMPLETED")
    print("-" * 40)
    print("Total rows:", len(df))
    print("Products:", df["id"].nunique())
    print("Features:", len(df.columns))
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Missing prices:", df["sell_price"].isna().sum())
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
