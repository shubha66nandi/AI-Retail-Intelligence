from sqlalchemy import Integer, Numeric, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base
from datetime import datetime


class Sale(Base):
    __tablename__ = "sales"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id"), nullable=False
    )

    quantity: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    unit_price: Mapped[float] = mapped_column(
        Numeric(10, 2), nullable=False
    )

    total_amount: Mapped[float] = mapped_column(
        Numeric(12, 2), nullable=False
    )

    sale_date: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )