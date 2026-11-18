from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.models import Inventory
from app.services.inventory_service import (
    calculate_forecast_reorder_quantity,
    get_forecast_stock_status,
)

from pathlib import Path

import joblib
import pandas as pd

from fastapi import APIRouter, HTTPException


router = APIRouter(
    prefix="/forecast",
    tags=["Demand Forecasting"],
)


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "postgres_daily_sales.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "demand_forecasting"
    / "postgres_hgb_model.joblib"
)

FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "day_of_week",
    "month",
]


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Forecasting model not found")

    return joblib.load(MODEL_FILE)


@router.get("/next-day")
def get_next_day_forecast():

    try:
        if not DATA_FILE.exists():
            raise FileNotFoundError("Daily sales dataset not found")

        model = load_model()

        df = pd.read_csv(
            DATA_FILE,
            parse_dates=["date"],
        )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No sales history available for forecasting",
            )

        latest_date = df["date"].max()
        forecast_date = latest_date + pd.Timedelta(days=1)

        forecast_rows = []

        for _, group in df.groupby(
            ["product_id", "sku", "product_name"]
        ):

            group = group.sort_values("date").reset_index(drop=True)

            if len(group) < 28:
                continue

            sales = group["units_sold"]

            features = {
                "lag_1": sales.iloc[-1],
                "lag_7": sales.iloc[-7],
                "lag_14": sales.iloc[-14],
                "lag_28": sales.iloc[-28],
                "rolling_mean_7": sales.tail(7).mean(),
                "rolling_mean_14": sales.tail(14).mean(),
                "rolling_mean_28": sales.tail(28).mean(),
                "rolling_std_7": sales.tail(7).std(),
                "rolling_std_28": sales.tail(28).std(),
                "day_of_week": forecast_date.dayofweek,
                "month": forecast_date.month,
            }

            feature_df = pd.DataFrame(
                [features],
                columns=FEATURES,
            )

            prediction = model.predict(feature_df)[0]

            forecast_rows.append({
                "product_id": int(group["product_id"].iloc[0]),
                "sku": group["sku"].iloc[0],
                "product_name": group["product_name"].iloc[0],
                "forecast_date": forecast_date.date().isoformat(),
                "predicted_units": round(
                    max(0, float(prediction)), 2
                ),
            })

        if not forecast_rows:
            raise HTTPException(
                status_code=404,
                detail="No products available for forecasting",
            )

        total_demand = sum(
            item["predicted_units"]
            for item in forecast_rows
        )

        return {
            "status": "success",
            "forecast_date": forecast_date.date().isoformat(),
            "products_forecasted": len(forecast_rows),
            "total_predicted_demand": round(total_demand, 2),
            "forecast_type": "synthetic_demo_data",
            "forecasts": forecast_rows,
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {str(error)}",
        )



@router.get("/inventory-decisions")
def get_forecast_inventory_decisions(
    db: Session = Depends(get_db),
):
    """
    Combine next-day demand forecasts with current inventory.
    This endpoint only recommends reorder quantities.
    It does not modify inventory.
    """

    # Reuse the existing forecasting endpoint logic.
    forecast_result = get_next_day_forecast()

    decisions = []

    for forecast in forecast_result["forecasts"]:

        inventory = (
            db.query(Inventory)
            .filter(
                Inventory.product_id == forecast["product_id"]
            )
            .first()
        )

        if inventory is None:
            decisions.append({
                **forecast,
                "current_stock": None,
                "stock_status": "inventory_record_missing",
                "recommended_reorder_quantity": None,
            })
            continue

        predicted_demand = forecast["predicted_units"]

        reorder_quantity = calculate_forecast_reorder_quantity(
            inventory,
            predicted_demand,
        )

        stock_status = get_forecast_stock_status(
            inventory,
            predicted_demand,
        )

        decisions.append({
            **forecast,
            "current_stock": inventory.quantity,
            "reorder_level": inventory.reorder_level,
            "target_stock": inventory.target_stock,
            "stock_status": stock_status,
            "recommended_reorder_quantity": reorder_quantity,
        })

    total_reorder_quantity = sum(
        item["recommended_reorder_quantity"] or 0
        for item in decisions
    )

    return {
        "status": "success",
        "forecast_date": forecast_result["forecast_date"],
        "products_evaluated": len(decisions),
        "products_requiring_reorder": sum(
            1
            for item in decisions
            if item["recommended_reorder_quantity"] is not None
            and item["recommended_reorder_quantity"] > 0
        ),
        "total_recommended_reorder_quantity": total_reorder_quantity,
        "decisions": decisions,
    }
