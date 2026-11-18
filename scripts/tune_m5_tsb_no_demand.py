from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

TRAIN = BASE / "data/processed/m5/m5_model_train.csv"
VAL = BASE / "data/processed/m5/m5_validation.csv"
PATTERNS = BASE / "models/demand_forecasting/m5_demand_patterns.csv"

ALPHA = 0.20
BETAS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]


def tsb(train, validation, alpha, beta):
    nonzero = train[train > 0]

    probability = len(nonzero) / len(train)
    size = float(nonzero[0]) if len(nonzero) else 0.0

    for value in nonzero[1:]:
        size += alpha * (value - size)

    predictions = []

    for actual in validation:
        predictions.append(probability * size)

        occurrence = float(actual > 0)
        probability += beta * (occurrence - probability)

        if occurrence:
            size += alpha * (actual - size)

    return np.array(predictions)


def main():
    train = pd.read_csv(TRAIN)
    val = pd.read_csv(VAL)
    patterns = pd.read_csv(PATTERNS)

    ids = set(
        patterns.loc[
            patterns["demand_pattern"] == "No Demand", "id"
        ]
    )

    train_groups = {
        key: group.sort_values("date")["units_sold"].to_numpy()
        for key, group in train[train.id.isin(ids)].groupby("id")
    }

    val_groups = {
        key: group.sort_values("date")["units_sold"].to_numpy()
        for key, group in val[val.id.isin(ids)].groupby("id")
    }

    results = []

    for beta in BETAS:
        abs_error = 0.0
        squared_error = 0.0
        actual_total = 0.0
        count = 0

        for product, actual in val_groups.items():
            predicted = tsb(
                train_groups[product],
                actual,
                ALPHA,
                beta,
            )

            errors = actual - predicted

            abs_error += np.abs(errors).sum()
            squared_error += np.square(errors).sum()
            actual_total += np.abs(actual).sum()
            count += len(actual)

        results.append({
            "Alpha": ALPHA,
            "Beta": beta,
            "MAE": abs_error / count,
            "RMSE": np.sqrt(squared_error / count),
            "WAPE": abs_error / actual_total * 100,
        })

    result = pd.DataFrame(results).sort_values("WAPE")

    print("\nTSB NO-DEMAND PARAMETER RESULTS")
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
