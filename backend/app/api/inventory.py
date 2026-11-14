from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.database.connection import get_db
from app.models import Inventory, Product
from app.schemas.inventory import InventoryCreate, InventoryUpdate, InventoryResponse


router = APIRouter(prefix="/inventory", tags=["Inventory"])


# Create inventory
@router.post("/", response_model=InventoryResponse, status_code=201)
def create_inventory(data: InventoryCreate, db: Session = Depends(get_db)):

    product = db.get(Product, data.product_id)

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    inventory = Inventory(**data.model_dump())
    db.add(inventory)

    try:
        db.commit()
        db.refresh(inventory)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Inventory already exists for this product",
        )

    return inventory


# Get all inventory
@router.get("/", response_model=list[InventoryResponse])
def get_inventory(db: Session = Depends(get_db)):
    return db.scalars(select(Inventory)).all()


# Get inventory by ID
@router.get("/{inventory_id}", response_model=InventoryResponse)
def get_inventory_by_id(inventory_id: int, db: Session = Depends(get_db)):

    inventory = db.get(Inventory, inventory_id)

    if not inventory:
        raise HTTPException(status_code=404, detail="Inventory not found")

    return inventory


# Update inventory
@router.put("/{inventory_id}", response_model=InventoryResponse)
def update_inventory(
    inventory_id: int,
    data: InventoryUpdate,
    db: Session = Depends(get_db),
):

    inventory = db.get(Inventory, inventory_id)

    if not inventory:
        raise HTTPException(status_code=404, detail="Inventory not found")

    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(inventory, key, value)

    db.commit()
    db.refresh(inventory)

    return inventory


# Delete inventory
@router.delete("/{inventory_id}", status_code=204)
def delete_inventory(inventory_id: int, db: Session = Depends(get_db)):

    inventory = db.get(Inventory, inventory_id)

    if not inventory:
        raise HTTPException(status_code=404, detail="Inventory not found")

    db.delete(inventory)
    db.commit()