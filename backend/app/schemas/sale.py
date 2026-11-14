from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class SaleCreate(BaseModel):
    product_id: int = Field(gt=0)
    quantity: int = Field(gt=0)
    unit_price: Decimal = Field(gt=0, max_digits=10, decimal_places=2)


class SaleUpdate(BaseModel):
    quantity: int | None = Field(default=None, gt=0)
    unit_price: Decimal | None = Field(
        default=None, gt=0, max_digits=10, decimal_places=2
    )


class SaleResponse(BaseModel):
    id: int
    product_id: int
    quantity: int
    unit_price: Decimal
    total_amount: Decimal
    sale_date: datetime

    model_config = ConfigDict(from_attributes=True)