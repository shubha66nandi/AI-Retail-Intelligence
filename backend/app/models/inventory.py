
from datetime import datetime

from sqlalchemy import Integer, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class Inventory(Base):
    __tablename__ = "inventory"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id"), unique=True, nullable=False
    )

    quantity: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    reorder_level: Mapped[int] = mapped_column(
        Integer, default=10, nullable=False
    )

    target_stock: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    last_updated: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )
