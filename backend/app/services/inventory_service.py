
import math

from app.models import Inventory


def calculate_reorder_quantity(inventory: Inventory) -> int:
    """
    Calculate the quantity required to reach target stock.
    """

    reorder_quantity = inventory.target_stock - inventory.quantity

    return max(0, reorder_quantity)


def is_low_stock(inventory: Inventory) -> bool:
    """
    Check whether current stock is at or below reorder level.
    """

    return inventory.quantity <= inventory.reorder_level


def calculate_forecast_reorder_quantity(
    inventory: Inventory,
    predicted_demand: float,
) -> int:
    """
    Calculate reorder quantity to cover forecast demand
    while maintaining target stock after expected sales.
    """

    forecast_required_stock = math.ceil(predicted_demand)

    required_stock = (
        forecast_required_stock + inventory.target_stock
    )

    reorder_quantity = required_stock - inventory.quantity

    return max(0, reorder_quantity)

def get_forecast_stock_status(
    inventory: Inventory,
    predicted_demand: float,
) -> str:
    """
    Determine stock status using predicted demand.
    """

    if inventory.quantity <= inventory.reorder_level:
        return "critical"

    if inventory.quantity < math.ceil(predicted_demand):
        return "insufficient_for_forecast"

    return "sufficient"
