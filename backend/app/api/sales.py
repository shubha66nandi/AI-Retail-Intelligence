from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.models import Sale, Product
from app.schemas.sale import SaleCreate, SaleUpdate, SaleResponse


router = APIRouter(prefix="/sales", tags=["Sales"])


# Create sale
@router.post("/", response_model=SaleResponse, status_code=201)
def create_sale(data: SaleCreate, db: Session = Depends(get_db)):

    product = db.get(Product, data.product_id)

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    sale = Sale(
        product_id=data.product_id,
        quantity=data.quantity,
        unit_price=data.unit_price,
        total_amount=data.quantity * data.unit_price,
    )

    db.add(sale)
    db.commit()
    db.refresh(sale)

    return sale


# Get all sales
@router.get("/", response_model=list[SaleResponse])
def get_sales(db: Session = Depends(get_db)):
    return db.scalars(select(Sale)).all()


# Get sale by ID
@router.get("/{sale_id}", response_model=SaleResponse)
def get_sale(sale_id: int, db: Session = Depends(get_db)):

    sale = db.get(Sale, sale_id)

    if not sale:
        raise HTTPException(status_code=404, detail="Sale not found")

    return sale


# Update sale
@router.put("/{sale_id}", response_model=SaleResponse)
def update_sale(
    sale_id: int,
    data: SaleUpdate,
    db: Session = Depends(get_db),
):

    sale = db.get(Sale, sale_id)

    if not sale:
        raise HTTPException(status_code=404, detail="Sale not found")

    updates = data.model_dump(exclude_unset=True)

    for key, value in updates.items():
        setattr(sale, key, value)

    sale.total_amount = sale.quantity * sale.unit_price

    db.commit()
    db.refresh(sale)

    return sale


# Delete sale
@router.delete("/{sale_id}", status_code=204)
def delete_sale(sale_id: int, db: Session = Depends(get_db)):

    sale = db.get(Sale, sale_id)

    if not sale:
        raise HTTPException(status_code=404, detail="Sale not found")

    db.delete(sale)
    db.commit()