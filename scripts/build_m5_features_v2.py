from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

INPUT = BASE / "data/processed/m5/ca1_foods_daily_v2.csv"
OUTPUT = BASE / "data/processed/m5/m5_features_v2.csv"

print("Loading V2 dataset...")

df = pd.read_csv(
    INPUT,
    parse_dates=["date"],
    dtype={
        "id": "category",
        "units_sold": "int16",
        "sell_price": "float32",
    },
)

df = df.sort_values(["id", "date"]).reset_index(drop=True)

grouped = df.groupby("id", observed=True)["units_sold"]

print("Creating lag features...")

for lag in [1, 7, 14, 21, 28, 56]:
    df[f"lag_{lag}"] = grouped.shift(lag).astype("float32")

print("Creating rolling features...")

for window in [7, 14, 28, 56]:
    df[f"rolling_mean_{window}"] = grouped.transform(
        lambda x: x.shift(1).rolling(window).mean()
    ).astype("float32")

for window in [7, 28]:
    df[f"rolling_std_{window}"] = grouped.transform(
        lambda x: x.shift(1).rolling(window).std()
    ).astype("float32")

print("Creating calendar features...")

df["day_of_week"] = df["date"].dt.day_of_week.astype("int8")
df["month"] = df["date"].dt.month.astype("int8")

df["snap_CA"] = df["snap_CA"].astype("int8")

for event_type in ["National", "Religious", "Cultural", "Sporting"]:
    df[f"event_{event_type.lower()}"] = (
        (df["event_type_1"] == event_type) |
        (df["event_type_2"] == event_type)
    ).astype("int8")

print("Creating price features...")

price_group = df.groupby("id", observed=True)["sell_price"]

previous_price = price_group.shift(1)
older_price = price_group.shift(2)

df["price_change_pct"] = (
    (previous_price - older_price)
    / older_price.replace(0, np.nan)
).astype("float32")

df["price_missing"] = df["sell_price"].isna().astype("int8")

print("Removing rows without 56-day history...")

required = [
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28", "lag_56",
    "rolling_mean_7", "rolling_mean_14",
    "rolling_mean_28", "rolling_mean_56",
    "rolling_std_7", "rolling_std_28",
]

df = df.dropna(subset=required).copy()

features = [
    "id", "date", "units_sold",
    *required,
    "day_of_week", "month", "snap_CA",
    "event_national", "event_religious",
    "event_cultural", "event_sporting",
    "sell_price", "price_missing", "price_change_pct",
]

df[features].to_csv(OUTPUT, index=False)

print("\nM5 V2 FEATURE ENGINEERING COMPLETED")
print("Rows:", len(df))
print("Products:", df["id"].nunique())
print("Columns:", len(features))
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Output:", OUTPUT)
