from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "data/processed/m5/m5_model_train.csv"
VALIDATION_FILE = BASE_DIR / "data/processed/m5/m5_validation.csv"

OUTPUT_FILE = (
    BASE_DIR / "models/demand_forecasting/m5_intermittent_results.csv"
)

ALPHAS = [0.05, 0.10, 0.20, 0.30]


def croston_forecast(train, validation, alpha, sba=False):
    values = train.astype(float)
    nonzero = np.flatnonzero(values > 0)

    if len(nonzero) == 0:
        return np.zeros(len(validation))

    first = nonzero[0]
    demand_size = values[first]
    interval = float(first + 1)
    last_demand = first

    for t in nonzero[1:]:
        gap = t - last_demand

        demand_size += alpha * (values[t] - demand_size)
        interval += alpha * (gap - interval)

        last_demand = t

    factor = (1 - alpha / 2) if sba else 1.0
    predictions = []

    for actual in validation:
        forecast = factor * demand_size / interval
        predictions.append(forecast)

        if actual > 0:
            gap = len(train) + len(predictions) - 1 - last_demand

            demand_size += alpha * (actual - demand_size)
            interval += alpha * (gap - interval)

            last_demand = len(train) + len(predictions) - 1

    return np.array(predictions)


def tsb_forecast(train, validation, alpha, beta):
    nonzero = train[train > 0]

    if len(nonzero) == 0:
        probability = 0.0
        demand_size = 0.0
    else:
        probability = len(nonzero) / len(train)
        demand_size = float(nonzero[0])

        for value in nonzero[1:]:
            demand_size += alpha * (value - demand_size)

    predictions = []

    for actual in validation:
        predictions.append(probability * demand_size)

        occurrence = 1.0 if actual > 0 else 0.0

        probability += beta * (occurrence - probability)

        if occurrence:
            demand_size += alpha * (actual - demand_size)

    return np.array(predictions)


def main():
    print("Loading training and validation data...")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VALIDATION_FILE)

    train_df["date"] = pd.to_datetime(train_df["date"])
    val_df["date"] = pd.to_datetime(val_df["date"])

    train_groups = {
        product: group.sort_values("date")["units_sold"].to_numpy()
        for product, group in train_df.groupby("id")
    }

    val_groups = {
        product: group.sort_values("date")["units_sold"].to_numpy()
        for product, group in val_df.groupby("id")
    }

    results = []

    print("Evaluating Croston, SBA and TSB...")

    for alpha in ALPHAS:
        for model_name in ["Croston", "SBA", "TSB"]:

            absolute_error = 0.0
            squared_error = 0.0
            actual_total = 0.0
            total_rows = 0

            for product, validation in val_groups.items():
                train = train_groups[product]

                if model_name == "Croston":
                    predictions = croston_forecast(
                        train, validation, alpha
                    )

                elif model_name == "SBA":
                    predictions = croston_forecast(
                        train, validation, alpha, sba=True
                    )

                else:
                    predictions = tsb_forecast(
                        train, validation, alpha, alpha
                    )

                errors = validation - predictions

                absolute_error += np.abs(errors).sum()
                squared_error += np.square(errors).sum()
                actual_total += np.abs(validation).sum()
                total_rows += len(validation)

            mae = absolute_error / total_rows
            rmse = np.sqrt(squared_error / total_rows)
            wape = absolute_error / actual_total * 100

            results.append({
                "Model": model_name,
                "Alpha": alpha,
                "MAE": mae,
                "RMSE": rmse,
                "WAPE": wape,
            })

            print(
                f"{model_name} | alpha={alpha:.2f} | "
                f"MAE={mae:.4f} | RMSE={rmse:.4f} | "
                f"WAPE={wape:.4f}%"
            )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("WAPE")

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nBEST CONFIGURATIONS")
    print(result_df.head(6).round(4).to_string(index=False))

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
