
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

INPUT = BASE / "data/processed/postgres_features.csv"

TRAIN_OUTPUT = BASE / "data/processed/postgres_train.csv"
TEST_OUTPUT = BASE / "data/processed/postgres_test.csv"

df = pd.read_csv(INPUT, parse_dates=["date"])

# Chronological split based on unique dates
dates = sorted(df["date"].unique())

split_index = int(len(dates) * 0.80)
cutoff_date = dates[split_index]

train = df[df["date"] < cutoff_date].copy()
test = df[df["date"] >= cutoff_date].copy()

TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

train.to_csv(TRAIN_OUTPUT, index=False)
test.to_csv(TEST_OUTPUT, index=False)

print("\nPOSTGRES TRAIN-TEST SPLIT COMPLETED")
print("=" * 45)

print("Training rows:", len(train))
print("Testing rows:", len(test))

print("Training period:",
      train["date"].min(), "to", train["date"].max())

print("Testing period:",
      test["date"].min(), "to", test["date"].max())

print("Training products:", train["product_id"].nunique())
print("Testing products:", test["product_id"].nunique())

print("\nTraining file:", TRAIN_OUTPUT)
print("Testing file:", TEST_OUTPUT)
