from pydantic import BaseModel, ConfigDict, Field


class ProductCreate(BaseModel):
    sku: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=150)
    category: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0)


class ProductResponse(ProductCreate):
    id: int
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class ProductUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=150)
    category: str | None = Field(default=None, max_length=100)
    price: float | None = Field(default=None, gt=0)
    is_active: bool | None = None