# backend/routers/customers.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime

from dependencies import get_db
from schemas import CustomerCreate, CustomerUpdate, ConvertWalkIn
from models import Customer, Sale, JarTracking, Reminder
from services.activity_service import update_customer_activity_status, update_all_customers_activity_status, mark_customer_inactive

router = APIRouter(prefix="/customers", tags=["Customers"])


@router.post("")
def create_customer(payload: CustomerCreate, db: Session = Depends(get_db)):
    existing = db.query(Customer).filter(func.lower(Customer.name) == func.lower(payload.name)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Customer name already exists")

    customer = Customer(
        name=payload.name.strip(),
        phone=payload.phone.strip() if payload.phone else None,
        address=payload.address,
        fixed_price_per_jar=payload.fixed_price_per_jar,
        delivery_type=payload.delivery_type
    )
    db.add(customer)
    db.commit()
    db.refresh(customer)

    # create default reminder when customer has delivery type
    if customer.delivery_type == "delivery":
        try:
            now = datetime.now()
            default_reminder = Reminder(
                customer_id=customer.id,
                custom_name=None,
                reason="delivery",
                frequency=3,
                next_date=now,
                note="Auto-created on profile creation (delivery).",
                status="pending"
            )
            db.add(default_reminder)
            db.commit()
        except Exception:
            db.rollback()

    return {"message": "Customer created", "customer": customer}


@router.get("")
def list_customers(activity_status: str = None, db: Session = Depends(get_db)):
    """
    List customers with optional activity_status filter.
    
    Valid filters: inactive, onetime, occasional, was_regular, active, no_pattern
    """
    query = db.query(Customer)
    
    if activity_status:
        query = query.filter(Customer.activity_status == activity_status)
    
    return query.order_by(Customer.name).all()


@router.put("/{customer_id}")
def update_customer(customer_id: int, payload: CustomerUpdate, db: Session = Depends(get_db)):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    existing = (
        db.query(Customer)
        .filter(func.lower(Customer.name) == func.lower(payload.name), Customer.id != customer_id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="A customer with this name already exists")

    customer.name = payload.name.strip()
    customer.phone = payload.phone.strip() if payload.phone else None
    customer.address = payload.address.strip() if payload.address else None
    customer.fixed_price_per_jar = payload.fixed_price_per_jar
    customer.delivery_type = payload.delivery_type or "self"

    db.commit()
    db.refresh(customer)
    return {"message": "Customer updated successfully", "customer": customer}


@router.delete("/{customer_id}")
def delete_customer(customer_id: int, db: Session = Depends(get_db)):
    total_due = (
        db.query(func.sum(Sale.due_amount))
        .filter(Sale.customer_id == customer_id)
        .scalar()
    ) or 0.0
    if total_due > 0:
        raise HTTPException(status_code=400, detail="Cannot delete customer with pending dues.")

    db.query(Customer).filter(Customer.id == customer_id).delete()
    db.commit()
    return {"message": "Customer deleted successfully"}


@router.post("/convert-walkin")
def convert_walkin(payload: ConvertWalkIn, db: Session = Depends(get_db)):
    """Convert a walk-in customer into a new profiled customer."""
    walkin_name = (payload.customer_name or "").strip()
    if not walkin_name:
        raise HTTPException(status_code=400, detail="customer_name (walk-in) required")

    new_name = (payload.name or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="name (new profile) required")

    existing_profile = db.query(Customer).filter(func.lower(Customer.name) == func.lower(new_name)).first()
    if existing_profile:
        raise HTTPException(status_code=400, detail="A profiled customer with this name already exists")

    new_customer = Customer(
        name=new_name,
        phone=payload.phone.strip() if payload.phone else None,
        address=payload.address,
        fixed_price_per_jar=payload.fixed_price_per_jar,
        delivery_type=payload.delivery_type,
        active=True
    )
    db.add(new_customer)
    db.commit()
    db.refresh(new_customer)

    updated_sales = (
        db.query(Sale)
        .filter(Sale.customer_id.is_(None), func.trim(Sale.customer_name) == walkin_name)
        .update(
            {
                Sale.customer_id: new_customer.id,
                Sale.customer_name: new_customer.name,
            },
            synchronize_session=False,
        )
    )

    jt = db.query(JarTracking).filter(JarTracking.customer_id.is_(None), func.trim(JarTracking.customer_name) == walkin_name).first()
    if jt:
        jt.customer_id = new_customer.id
        jt.customer_name = new_customer.name
        db.add(jt)
    else:
        total_given = (
            db.query(func.coalesce(func.sum(Sale.total_jars - Sale.customer_own_jars), 0))
            .filter(Sale.customer_id == new_customer.id)
            .scalar()
        )

        total_returned = (
            db.query(func.coalesce(func.sum(Sale.our_jars_returned), 0))
            .filter(Sale.customer_id == new_customer.id)
            .scalar()
        )

        current_due = total_given - total_returned

        jt = db.query(JarTracking).filter(JarTracking.customer_id == new_customer.id).first()

        if jt:
            jt.our_jars_given = total_given
            jt.our_jars_returned = total_returned
            jt.current_due_jars = current_due
        else:
            jt = JarTracking(
                customer_id=new_customer.id,
                our_jars_given=total_given,
                our_jars_returned=total_returned,
                current_due_jars=current_due,
            )
            db.add(jt)

    db.commit()

    return {
        "message": f"Walk-in '{walkin_name}' converted to profiled '{new_customer.name}' successfully.",
        "customer_id": new_customer.id,
        "updated_sales": int(updated_sales),
    }


@router.get("/check-name")
def check_customer_name(name: str, db: Session = Depends(get_db)):
    clean = name.strip().lower()
    exists = (
        db.query(Customer)
        .filter(func.lower(Customer.name) == clean)
        .first()
    )
    return {"exists": bool(exists)}


@router.post("/update-activity-status")
def update_all_activity_statuses(db: Session = Depends(get_db)):
    """
    Update activity status for all customers based on their purchase patterns.
    This should be run after migration or periodically to refresh statuses.
    """
    status_counts = update_all_customers_activity_status(db)
    return {
        "message": "Activity statuses updated for all customers",
        "summary": status_counts
    }


@router.post("/{customer_id}/mark-inactive")
def mark_inactive(customer_id: int, db: Session = Depends(get_db)):
    """
    Manually mark a customer as inactive.
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    success = mark_customer_inactive(customer_id, db)
    if success:
        return {"message": f"Customer {customer.name} marked as inactive"}
    else:
        raise HTTPException(status_code=500, detail="Failed to mark customer as inactive")

