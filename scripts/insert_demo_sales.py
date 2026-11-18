
from pathlib import Path

import pandas as pd
from sqlalchemy import select, func

from app.database.connection import SessionLocal
from app.models import Product, Sale, Inventory


BASE = Path(__file__).resolve().parents[1]
CSV_FILE = BASE / "data/processed/demo_sales.csv"

INITIAL_STOCK = 100
REORDER_LEVEL = 10
TARGET_STOCK = 50


def main():
    df = pd.read_csv(
        CSV_FILE,
        parse_dates=["sale_date"]
    )

    db = SessionLocal()

    try:
        # 1. Create or find demo products
        product_map = {}

        for row in df[
            ["sku", "product_name", "category", "unit_price"]
        ].drop_duplicates("sku").itertuples(index=False):

            sku, name, category, price = row

            product = db.scalars(
                select(Product).where(Product.sku == sku)
            ).first()

            if not product:
                product = Product(
                    sku=sku,
                    name=name,
                    category=category,
                    price=price,
                    is_active=True,
                )
                db.add(product)
                db.flush()

            product_map[sku] = product

        print("Products ready:", len(product_map))

        # 2. Create inventory records if missing
        for product in product_map.values():

            inventory = db.scalars(
                select(Inventory).where(
                    Inventory.product_id == product.id
                )
            ).first()

            if not inventory:
                db.add(
                    Inventory(
                        product_id=product.id,
                        quantity=INITIAL_STOCK,
                        reorder_level=REORDER_LEVEL,
                        target_stock=TARGET_STOCK,
                    )
                )

        db.flush()

        # 3. Check existing demo sales
        product_ids = [
            product.id for product in product_map.values()
        ]

        existing_sales = db.scalar(
            select(func.count(Sale.id)).where(
                Sale.product_id.in_(product_ids)
            )
        )

        if existing_sales == 0:

            sales = []

            for row in df.itertuples(index=False):
                product = product_map[row.sku]

                sales.append(
                    Sale(
                        product_id=product.id,
                        quantity=int(row.quantity),
                        unit_price=float(row.unit_price),
                        total_amount=float(row.total_amount),
                        sale_date=row.sale_date.to_pydatetime(),
                    )
                )

            db.add_all(sales)
            print("Sales records to insert:", len(sales))

        elif existing_sales == len(df):

            print("Demo sales already exist. Skipping insertion.")

        else:
            raise RuntimeError(
                f"Found {existing_sales} existing demo sales. "
                "Partial data detected. Stopping to avoid duplicates."
            )

        db.commit()

        print("\nDEMO DATA INSERTION SUCCESSFUL")
        print("Products:", len(product_map))
        print("Sales records:", len(df) if existing_sales == 0 else existing_sales)
        print("Inventory records ready.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
