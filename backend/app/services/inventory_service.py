
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
