from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

TRAIN = BASE / "data/processed/m5/m5_model_train.csv"
VAL = BASE / "data/processed/m5/m5_validation.csv"
PATTERNS = BASE / "models/demand_forecasting/m5_demand_patterns.csv"
MODEL = BASE / "models/demand_forecasting/m5_validation_model.joblib"

FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "day_of_week", "month", "sell_price", "price_missing",
]

ALPHA = 0.20
BETA = 0.30


def tsb(train, validation):
    nonzero = train[train > 0]

    probability = len(nonzero) / len(train)
    size = float(nonzero[0]) if len(nonzero) else 0.0

    for value in nonzero[1:]:
        size += ALPHA * (value - size)

    predictions = []

    for actual in validation:
        predictions.append(probability * size)

        occurrence = float(actual > 0)
        probability += BETA * (occurrence - probability)

        if occurrence:
            size += ALPHA * (actual - size)

    return np.array(predictions)


def metrics(actual, predicted):
    error = actual - predicted

    return {
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt(np.square(error).mean()),
        "WAPE": np.abs(error).sum() / np.abs(actual).sum() * 100,
    }


def main():
    print("Loading validation data...")

    train = pd.read_csv(TRAIN)
    val = pd.read_csv(VAL)
    patterns = pd.read_csv(PATTERNS)
    model = joblib.load(MODEL)

    train_groups = {
        key: group.sort_values("date")["units_sold"].to_numpy()
        for key, group in train.groupby("id")
    }

    val = val.sort_values(["id", "date"]).copy()

    val["HGB"] = model.predict(
        val[FEATURES].astype("float32")
    )

    val["Hybrid"] = val["HGB"].to_numpy()

    pattern_map = patterns.set_index("id")["demand_pattern"]

    val["demand_pattern"] = val["id"].map(pattern_map)

    no_demand_indices = val.index[
        val["demand_pattern"] == "No Demand"
    ]

    for product, indices in val.loc[no_demand_indices].groupby("id").groups.items():
        actual = val.loc[indices, "units_sold"].to_numpy()

        val.loc[indices, "Hybrid"] = tsb(
            train_groups[product],
            actual,
        )

    actual = val["units_sold"].to_numpy(dtype=float)

    print("\nVALIDATION COMPARISON")
    print("=" * 55)

    for name, prediction in [
        ("HistGradientBoosting", val["HGB"].to_numpy()),
        ("Improved Hybrid", val["Hybrid"].to_numpy()),
    ]:
        result = metrics(actual, prediction)

        print(f"\n{name}")
        for key, value in result.items():
            print(f"{key}: {value:.4f}")

    output = BASE / "models/demand_forecasting/m5_improved_hybrid_validation.csv"

    val[["id", "date", "units_sold", "HGB", "Hybrid", "demand_pattern"]].to_csv(
        output, index=False
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
