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
BETA = 0.20


def tsb(train, validation):
    nonzero = train[train > 0]

    if len(nonzero) == 0:
        probability = 0.0
        size = 0.0
    else:
        probability = len(nonzero) / len(train)
        size = float(nonzero[0])

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


def main():
    print("Loading data and model...")

    train_df = pd.read_csv(TRAIN)
    val_df = pd.read_csv(VAL)
    patterns = pd.read_csv(PATTERNS)
    model = joblib.load(MODEL)

    train_groups = {
        key: group.sort_values("date")["units_sold"].to_numpy()
        for key, group in train_df.groupby("id")
    }

    val_df = val_df.sort_values(["id", "date"])

    val_df["HGB"] = model.predict(
        val_df[FEATURES].astype("float32")
    )

    val_df["TSB"] = 0.0

    for product, indices in val_df.groupby("id").groups.items():
        actual = val_df.loc[indices, "units_sold"].to_numpy()

        val_df.loc[indices, "TSB"] = tsb(
            train_groups[product],
            actual,
        )

    val_df = val_df.merge(
        patterns[["id", "demand_pattern"]],
        on="id",
        validate="many_to_one",
    )

    val_df["Baseline"] = val_df["rolling_mean_7"]

    results = []

    for pattern, group in val_df.groupby("demand_pattern"):
        actual = group["units_sold"].to_numpy(dtype=float)
        total = np.abs(actual).sum()

        for name in ["HGB", "TSB", "Baseline"]:
            pred = group[name].to_numpy(dtype=float)
            error = np.abs(actual - pred)

            results.append({
                "Pattern": pattern,
                "Model": name,
                "MAE": error.mean(),
                "RMSE": np.sqrt(np.mean((actual - pred) ** 2)),
                "WAPE": error.sum() / total * 100 if total else np.nan,
            })

    result = pd.DataFrame(results)

    print("\nPATTERN-WISE MODEL COMPARISON")
    print(result.round(4).to_string(index=False))

    output = BASE / "models/demand_forecasting/m5_pattern_model_comparison.csv"
    result.to_csv(output, index=False)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
