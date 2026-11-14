from datetime import datetime
from pydantic import BaseModel, Field


class InventoryUpdate(BaseModel):
    warehouse_stock: int | None = Field(default=None, ge=0)
    shelf_stock: int | None = Field(default=None, ge=0)
    shelf_capacity: int | None = Field(default=None, ge=1)


class InventoryResponse(BaseModel):
    id: int
    product_id: int
    warehouse_stock: int
    shelf_stock: int
    total_stock: int
    shelf_capacity: int
    status: str
    last_updated: datetime

    model_config = {"from_attributes": True}