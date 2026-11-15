
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.database.connection import get_db
from app.models import Inventory, Product
from app.schemas.inventory import (
    InventoryCreate,
    InventoryUpdate,
    InventoryResponse,
)

from app.services.inventory_service import (
    calculate_reorder_quantity,
    is_low_stock,
)


router = APIRouter(prefix="/inventory", tags=["Inventory"])


# Create inventory
@router.post("/", response_model=InventoryResponse, status_code=201)
def create_inventory(data: InventoryCreate, db: Session = Depends(get_db)):

    product = db.get(Product, data.product_id)

    if not product:
        raise HTTPException(
            status_code=404,
            detail="Product not found",
        )

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


# Get low-stock inventory
@router.get("/low-stock", response_model=list[InventoryResponse])
def get_low_stock(db: Session = Depends(get_db)):

    statement = select(Inventory).where(
        Inventory.quantity <= Inventory.reorder_level
    )

    return db.scalars(statement).all()


# Get reorder suggestions
@router.get("/reorder-suggestions")
def get_reorder_suggestions(db: Session = Depends(get_db)):

    inventories = db.scalars(select(Inventory)).all()

    suggestions = []

    for inventory in inventories:

        if is_low_stock(inventory):

            reorder_quantity = calculate_reorder_quantity(inventory)

            suggestions.append({
                "product_id": inventory.product_id,
                "current_stock": inventory.quantity,
                "reorder_level": inventory.reorder_level,
                "target_stock": inventory.target_stock,
                "reorder_quantity": reorder_quantity,
                "status": "Low Stock",
            })

    return suggestions


# Get inventory by ID
@router.get("/{inventory_id}", response_model=InventoryResponse)
def get_inventory_by_id(inventory_id: int, db: Session = Depends(get_db)):

    inventory = db.get(Inventory, inventory_id)

    if not inventory:
        raise HTTPException(
            status_code=404,
            detail="Inventory not found",
        )

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
        raise HTTPException(
            status_code=404,
            detail="Inventory not found",
        )

    updates = data.model_dump(exclude_unset=True)

    for key, value in updates.items():
        setattr(inventory, key, value)

    try:
        db.commit()
        db.refresh(inventory)

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to update inventory",
        )

    return inventory


# Delete inventory
@router.delete("/{inventory_id}", status_code=204)
def delete_inventory(inventory_id: int, db: Session = Depends(get_db)):

    inventory = db.get(Inventory, inventory_id)

    if not inventory:
        raise HTTPException(
            status_code=404,
            detail="Inventory not found",
        )

    try:
        db.delete(inventory)
        db.commit()

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to delete inventory",
        )
