
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from app.database.connection import engine


BASE = Path(__file__).resolve().parents[1]

OUTPUT = BASE / "data/processed/postgres_daily_sales.csv"


def main():

    print("Loading demo products and sales from PostgreSQL...")

    with engine.connect() as connection:

        products = pd.read_sql(
            text("""
                SELECT id AS product_id, sku, name AS product_name, category
                FROM products
                WHERE sku LIKE 'DEMO-%'
                ORDER BY id
            """),
            connection,
        )

        sales = pd.read_sql(
            text("""
                SELECT
                    product_id,
                    sale_date::date AS date,
                    SUM(quantity) AS units_sold
                FROM sales
                WHERE product_id IN (
                    SELECT id
                    FROM products
                    WHERE sku LIKE 'DEMO-%'
                )
                GROUP BY product_id, sale_date::date
                ORDER BY product_id, date
            """),
            connection,
        )

    if products.empty or sales.empty:
        raise ValueError("No demo products or sales found.")

    sales["date"] = pd.to_datetime(sales["date"])

    start_date = sales["date"].min()
    end_date = sales["date"].max()

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    print("Creating complete product-date combinations...")

    calendar = pd.DataFrame({"date": dates})

    products["key"] = 1
    calendar["key"] = 1

    daily = products.merge(calendar, on="key").drop(columns="key")

    daily = daily.merge(
        sales,
        on=["product_id", "date"],
        how="left",
    )

    daily["units_sold"] = (
        daily["units_sold"]
        .fillna(0)
        .astype("int32")
    )

    daily = daily.sort_values(
        ["product_id", "date"]
    ).reset_index(drop=True)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(OUTPUT, index=False)

    print("\nPOSTGRES DAILY SALES DATASET CREATED")
    print("=" * 45)

    print("Products:", daily["product_id"].nunique())
    print("Date range:", start_date.date(), "to", end_date.date())
    print("Days:", len(dates))
    print("Total rows:", len(daily))
    print("Total units sold:", daily["units_sold"].sum())
    print("Zero-sales rows:", (daily["units_sold"] == 0).sum())

    print("\nOutput:", OUTPUT)


if __name__ == "__main__":
    main()
