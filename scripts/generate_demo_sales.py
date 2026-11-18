
from pathlib import Path
import numpy as np
import pandas as pd

# ==================== CONFIGURATION ====================

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / "data/processed/demo_sales.csv"

rng = np.random.default_rng(42)

# ==================== PRODUCT DATA ====================

products = [
    ("Keyboard", "Electronics", 850, 8),
    ("Mouse", "Electronics", 450, 12),
    ("Headphones", "Electronics", 1200, 5),
    ("USB Cable", "Electronics", 180, 20),
    ("Power Bank", "Electronics", 1500, 4),
    ("Notebook", "Stationery", 60, 25),
    ("Pen", "Stationery", 15, 40),
    ("Marker", "Stationery", 35, 18),
    ("Water Bottle", "Lifestyle", 250, 15),
    ("Backpack", "Lifestyle", 950, 5),
    ("T-Shirt", "Fashion", 500, 10),
    ("Jeans", "Fashion", 1200, 4),
    ("Socks", "Fashion", 100, 20),
    ("Rice 5kg", "Grocery", 350, 18),
    ("Cooking Oil", "Grocery", 180, 22),
    ("Biscuits", "Grocery", 30, 35),
    ("Shampoo", "Personal Care", 220, 12),
    ("Soap", "Personal Care", 45, 30),
    ("Toothpaste", "Personal Care", 110, 16),
    ("Detergent", "Household", 280, 9),
]

# ==================== DATE RANGE ====================

start_date = pd.Timestamp("2026-01-01")
dates = pd.date_range(
    start=start_date,
    periods=180,
    freq="D"
)

records = []

# ==================== GENERATE SALES ====================

for index, (name, category, price, base_demand) in enumerate(
    products, start=1
):

    for date in dates:

        # Weekend demand variation
        weekday_factor = (
            1.25 if date.day_of_week >= 5 else 1.0
        )

        # Gradual demand trend
        trend_factor = (
            1 + 0.0008 * (date - start_date).days
        )

        expected_demand = (
            base_demand * weekday_factor * trend_factor
        )

        quantity = int(rng.poisson(expected_demand))

        if quantity <= 0:
            continue

        # Split daily demand into transactions
        remaining = quantity

        while remaining > 0:

            sold = int(
                rng.integers(1, min(remaining, 5) + 1)
            )

            remaining -= sold

            sale_time = date + pd.Timedelta(
                seconds=int(
                    rng.integers(8 * 3600, 21 * 3600)
                ),
                microseconds=len(records)
            )

            records.append({
                "sku": f"DEMO-{index:03d}",
                "product_name": name,
                "category": category,
                "quantity": sold,
                "unit_price": price,
                "total_amount": round(sold * price, 2),
                "sale_date": sale_time,
            })

# ==================== CREATE DATAFRAME ====================

df = pd.DataFrame(records)

df = df.sort_values(
    "sale_date"
).reset_index(drop=True)

# ==================== SAVE CSV ====================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

# ==================== VALIDATION SUMMARY ====================

print("\nDEMO DATA GENERATED")
print("=" * 45)

print("Products:", df["sku"].nunique())
print("Sales records:", len(df))
print("Total units sold:", df["quantity"].sum())

print(
    "Date range:",
    df["sale_date"].min(),
    "to",
    df["sale_date"].max()
)

print("Output:", OUTPUT)

print("\nSales records by category:")
print(
    df.groupby("category")["quantity"]
    .sum()
    .sort_values(ascending=False)
)
