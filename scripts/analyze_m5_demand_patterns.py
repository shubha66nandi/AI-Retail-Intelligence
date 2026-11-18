from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data/processed/m5/m5_model_train.csv"
OUTPUT_FILE = BASE_DIR / "models/demand_forecasting/m5_demand_patterns.csv"

CHUNK_SIZE = 100_000

stats = {}

print("Analyzing product demand patterns...")

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["id", "units_sold"],
    chunksize=CHUNK_SIZE,
):
    grouped = chunk.groupby("id")["units_sold"].agg(
        total_days="count",
        nonzero_days=lambda x: (x > 0).sum(),
        positive_sum=lambda x: x[x > 0].sum(),
        positive_sum_sq=lambda x: np.square(x[x > 0]).sum(),
    )

    for product_id, row in grouped.iterrows():
        if product_id not in stats:
            stats[product_id] = {
                "total_days": 0,
                "nonzero_days": 0,
                "positive_sum": 0.0,
                "positive_sum_sq": 0.0,
            }

        item = stats[product_id]
        item["total_days"] += int(row["total_days"])
        item["nonzero_days"] += int(row["nonzero_days"])
        item["positive_sum"] += float(row["positive_sum"])
        item["positive_sum_sq"] += float(row["positive_sum_sq"])

results = []

for product_id, item in stats.items():
    days = item["total_days"]
    nonzero = item["nonzero_days"]
    total = item["positive_sum"]
    sum_sq = item["positive_sum_sq"]

    if nonzero == 0:
        adi = np.inf
        cv_squared = np.nan
        category = "No Demand"
    else:
        mean_demand = total / nonzero
        variance = max(
            (sum_sq / nonzero) - mean_demand ** 2,
            0.0,
        )

        adi = days / nonzero
        cv_squared = variance / (mean_demand ** 2)

        if adi < 1.32 and cv_squared < 0.49:
            category = "Smooth"
        elif adi < 1.32 and cv_squared >= 0.49:
            category = "Erratic"
        elif adi >= 1.32 and cv_squared < 0.49:
            category = "Intermittent"
        else:
            category = "Lumpy"

    results.append({
        "id": product_id,
        "total_days": days,
        "nonzero_days": nonzero,
        "ADI": adi,
        "CV_squared": cv_squared,
        "demand_pattern": category,
    })

df = pd.DataFrame(results)

df.to_csv(OUTPUT_FILE, index=False)

print("\nDEMAND PATTERN SUMMARY")
print("=" * 55)

summary = df.groupby("demand_pattern").agg(
    Products=("id", "count"),
    Average_ADI=("ADI", "mean"),
    Average_CV_squared=("CV_squared", "mean"),
)

print(summary.round(3).to_string())

print("\nTotal products:", len(df))
print("Saved:", OUTPUT_FILE)
