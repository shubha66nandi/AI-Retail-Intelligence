from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw" / "m5"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "m5"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading M5 sales data...")

sales = pd.read_csv(DATA_DIR / "sales_train_validation.csv")

sample = sales[
    (sales["store_id"] == "CA_1") &
    (sales["cat_id"] == "FOODS")
].copy()

print("Selected products:", len(sample))

id_columns = [
    "id", "item_id", "dept_id",
    "cat_id", "store_id", "state_id"
]

daily_columns = [
    col for col in sample.columns
    if col.startswith("d_")
]

long_data = sample.melt(
    id_vars=id_columns,
    value_vars=daily_columns,
    var_name="d",
    value_name="units_sold"
)

print("Loading calendar data...")

calendar = pd.read_csv(DATA_DIR / "calendar.csv")

calendar_columns = [
    "d",
    "date",
    "wm_yr_wk",
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
]

df = long_data.merge(
    calendar[calendar_columns],
    on="d",
    how="left",
    validate="many_to_one",
)

print("Joining selling prices...")

prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

df = df.merge(
    prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left",
    validate="many_to_one",
)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["id", "date"]).reset_index(drop=True)

output_file = OUTPUT_DIR / "ca1_foods_daily_v2.csv"

df.to_csv(output_file, index=False)

print("\nM5 V2 PREPARATION COMPLETED")
print("Rows:", len(df))
print("Products:", df["id"].nunique())
print("SNAP CA days:", df["snap_CA"].sum())
print("Event 1 rows:", df["event_type_1"].notna().sum())
print("Event 2 rows:", df["event_type_2"].notna().sum())
print("Missing prices:", df["sell_price"].isna().sum())
print("Output:", output_file)
