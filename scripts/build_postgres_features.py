
from pathlib import Path

import pandas as pd


# ================= CONFIGURATION =================

BASE = Path(__file__).resolve().parents[1]

INPUT = BASE / "data/processed/postgres_daily_sales.csv"
OUTPUT = BASE / "data/processed/postgres_features.csv"

# ================= LOAD DATA =================

print("Loading PostgreSQL daily sales data...")

df = pd.read_csv(
    INPUT,
    parse_dates=["date"],
)

df = df.sort_values(
    ["product_id", "date"]
).reset_index(drop=True)

grouped = df.groupby("product_id")["units_sold"]

# ================= LAG FEATURES =================

print("Creating lag features...")

for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = (
        grouped.shift(lag).astype("float32")
    )

# ================= ROLLING FEATURES =================

print("Creating rolling mean features...")

for window in [7, 14, 28]:
    df[f"rolling_mean_{window}"] = (
        grouped.transform(
            lambda x: x.shift(1).rolling(
                window,
                min_periods=window,
            ).mean()
        ).astype("float32")
    )

print("Creating rolling standard deviation...")

for window in [7, 28]:
    df[f"rolling_std_{window}"] = (
        grouped.transform(
            lambda x: x.shift(1).rolling(
                window,
                min_periods=window,
            ).std()
        ).astype("float32")
    )

# ================= CALENDAR FEATURES =================

print("Creating calendar features...")

df["day_of_week"] = (
    df["date"].dt.day_of_week.astype("int8")
)

df["month"] = (
    df["date"].dt.month.astype("int8")
)

# ================= REMOVE INCOMPLETE HISTORY =================

required_features = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
]

print("Removing rows without sufficient history...")

df = df.dropna(
    subset=required_features
).copy()

# ================= FINAL COLUMNS =================

features = [
    "product_id",
    "sku",
    "product_name",
    "category",
    "date",
    "units_sold",
    *required_features,
    "day_of_week",
    "month",
]

df = df[features]

# ================= SAVE DATA =================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT,
    index=False,
)

# ================= SUMMARY =================

print("\nPOSTGRES FEATURE ENGINEERING COMPLETED")
print("=" * 50)

print("Products:", df["product_id"].nunique())
print("Rows:", len(df))
print("Features:", len(required_features) + 2)
print("Date range:", df["date"].min(), "to", df["date"].max())

print("\nFeature columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nOutput:", OUTPUT)
