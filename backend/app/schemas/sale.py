from datetime import date, datetime
from pydantic import BaseModel, Field


class SaleCreate(BaseModel):
    product_id: int
    sale_date: date
    quantity_sold: int = Field(..., gt=0)


class SaleResponse(BaseModel):
    id: int
    product_id: int
    sale_date: date
    quantity_sold: int
    total_revenue: float
    created_at: datetime

    model_config = {"from_attributes": True}