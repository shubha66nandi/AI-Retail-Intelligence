
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "data/processed/m5/m5_train.csv"
RESULT_FILE = BASE_DIR / "models/demand_forecasting/m5_model_vs_baseline.csv"
OUTPUT_FILE = BASE_DIR / "models/demand_forecasting/m5_training_tier_comparison.csv"

CHUNK_SIZE = 200_000


def main():
    print("Calculating training-period sales by product...")

    sales_by_product = {}

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=["id", "units_sold"],
        chunksize=CHUNK_SIZE,
    ):
        grouped = chunk.groupby("id")["units_sold"].sum()

        for product_id, sales in grouped.items():
            sales_by_product[product_id] = (
                sales_by_product.get(product_id, 0) + sales
            )

    training = pd.DataFrame({
        "product_id": list(sales_by_product.keys()),
        "training_sales": list(sales_by_product.values()),
    })

    training["demand_tier"] = pd.qcut(
        training["training_sales"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
    )

    print("Loading model comparison results...")

    results = pd.read_csv(RESULT_FILE)

    df = results.merge(
        training,
        on="product_id",
        how="inner",
        validate="one_to_one",
    )

    # Reconstruct absolute errors from product WAPE.
    # WAPE values in the source CSV are rounded to 4 decimals.
    df["ML_error"] = df["ML_WAPE"] * df["total_actual_sales"] / 100
    df["Baseline_error"] = (
        df["Baseline_WAPE"] * df["total_actual_sales"] / 100
    )

    summary = df.groupby(
        "demand_tier", observed=True
    ).apply(
        lambda group: pd.Series({
            "Products": len(group),
            "Training_Sales": group["training_sales"].sum(),
            "ML_MAE": group["ML_MAE"].mean(),
            "Baseline_MAE": group["Baseline_MAE"].mean(),
            "ML_WAPE": (
                group["ML_error"].sum()
                / group["total_actual_sales"].sum() * 100
            ),
            "Baseline_WAPE": (
                group["Baseline_error"].sum()
                / group["total_actual_sales"].sum() * 100
            ),
            "Products_ML_Better": (
                group["ML_better_MAE"].sum()
            ),
        }),
        include_groups=False,
    )

    summary.to_csv(OUTPUT_FILE)

    print("\nTRAINING-BASED DEMAND TIER RESULTS")
    print("=" * 65)
    print(summary.round(3).to_string())

    print("\nProducts matched:", len(df))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
