
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data/processed/m5/m5_features.csv"
OUTPUT_DIR = BASE_DIR / "data/processed/m5"

TRAIN_FILE = OUTPUT_DIR / "m5_train.csv"
TEST_FILE = OUTPUT_DIR / "m5_test.csv"

SPLIT_DATE = pd.Timestamp("2015-04-14")
CHUNK_SIZE = 200_000


def main():
    print("Creating chronological train/test split...")

    # Remove previous outputs to avoid appending to old results.
    for file in [TRAIN_FILE, TEST_FILE]:
        if file.exists():
            file.unlink()

    train_rows = 0
    test_rows = 0
    first_train = None
    last_train = None
    first_test = None
    last_test = None

    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        parse_dates=["date"],
    ):
        train = chunk[chunk["date"] < SPLIT_DATE]
        test = chunk[chunk["date"] >= SPLIT_DATE]

        if not train.empty:
            train.to_csv(
                TRAIN_FILE,
                mode="a",
                index=False,
                header=not TRAIN_FILE.exists(),
            )
            train_rows += len(train)
            first_train = min(first_train, train["date"].min()) if first_train else train["date"].min()
            last_train = max(last_train, train["date"].max()) if last_train else train["date"].max()

        if not test.empty:
            test.to_csv(
                TEST_FILE,
                mode="a",
                index=False,
                header=not TEST_FILE.exists(),
            )
            test_rows += len(test)
            first_test = min(first_test, test["date"].min()) if first_test else test["date"].min()
            last_test = max(last_test, test["date"].max()) if last_test else test["date"].max()

    print("\nCHRONOLOGICAL SPLIT COMPLETED")
    print("-" * 40)
    print("Training rows:", train_rows)
    print("Testing rows:", test_rows)
    print("Training dates:", first_train, "to", last_train)
    print("Testing dates:", first_test, "to", last_test)
    print("Training file:", TRAIN_FILE)
    print("Testing file:", TEST_FILE)


if __name__ == "__main__":
    main()
