from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data/processed/m5/m5_features_v2.csv"
OUTPUT_DIR = BASE_DIR / "data/processed/m5"

TRAIN_FILE = OUTPUT_DIR / "m5_v2_model_train.csv"
VALIDATION_FILE = OUTPUT_DIR / "m5_v2_validation.csv"

SPLIT_DATE = pd.Timestamp("2014-06-16")
VALIDATION_END = pd.Timestamp("2015-04-13")

CHUNK_SIZE = 100_000


def main():
    print("Creating V2 training and validation datasets...")

    for file in [TRAIN_FILE, VALIDATION_FILE]:
        if file.exists():
            file.unlink()

    train_rows = 0
    validation_rows = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        parse_dates=["date"],
    ):
        train = chunk[chunk["date"] < SPLIT_DATE]

        validation = chunk[
            (chunk["date"] >= SPLIT_DATE) &
            (chunk["date"] <= VALIDATION_END)
        ]

        if not train.empty:
            train.to_csv(
                TRAIN_FILE,
                mode="a",
                index=False,
                header=not TRAIN_FILE.exists(),
            )
            train_rows += len(train)

        if not validation.empty:
            validation.to_csv(
                VALIDATION_FILE,
                mode="a",
                index=False,
                header=not VALIDATION_FILE.exists(),
            )
            validation_rows += len(validation)

    print("\nV2 VALIDATION SPLIT COMPLETED")
    print("=" * 50)
    print("Training rows:", train_rows)
    print("Validation rows:", validation_rows)
    print("Training file:", TRAIN_FILE)
    print("Validation file:", VALIDATION_FILE)


if __name__ == "__main__":
    main()
