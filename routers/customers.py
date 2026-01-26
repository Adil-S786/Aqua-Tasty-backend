# backend/routers/customers.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime

from dependencies import get_db
from schemas import CustomerCreate, CustomerUpdate, ConvertWalkIn
from models import Customer, Sale, JarTracking, Reminder, PaymentHistory
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
    Returns customers with their individual total_due (NOT combined for linked accounts).
    
    Valid filters: inactive, onetime, occasional, was_regular, active, no_pattern
    """
    query = db.query(Customer)
    
    if activity_status:
        query = query.filter(Customer.activity_status == activity_status)
    
    customers = query.order_by(Customer.name).all()
    
    # Add individual total_due for each customer
    result = []
    for customer in customers:
        # Get total due for THIS customer only (not linked accounts)
        total_due = (
            db.query(func.coalesce(func.sum(Sale.due_amount), 0.0))
            .filter(Sale.customer_id == customer.id)
            .scalar()
        )
        
        # Convert customer to dict and add total_due
        customer_dict = {
            "id": customer.id,
            "name": customer.name,
            "phone": customer.phone,
            "address": customer.address,
            "fixed_price_per_jar": customer.fixed_price_per_jar,
            "delivery_type": customer.delivery_type,
            "activity_status": customer.activity_status,
            "advance_payment": customer.advance_payment,
            "parent_customer_id": customer.parent_customer_id,
            "total_due": float(total_due)  # Individual due, not combined
        }
        result.append(customer_dict)
    
    return result


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



@router.post("/{customer_id}/link")
def link_customer_account(customer_id: int, parent_id: int, db: Session = Depends(get_db)):
    """
    Link a customer account to a parent account (e.g., shop to home).
    Only parent accounts (those without parent_customer_id) can have children.
    
    When linking:
    1. If either account has advance payment, use it to settle dues across both accounts (FIFO)
    2. Store any remaining advance in the parent account
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    parent = db.query(Customer).filter(Customer.id == parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent customer not found")
    
    # Prevent linking to a child account
    if parent.parent_customer_id is not None:
        raise HTTPException(status_code=400, detail="Cannot link to a child account. Please link to the parent account instead.")
    
    # Prevent self-linking
    if customer_id == parent_id:
        raise HTTPException(status_code=400, detail="Cannot link a customer to itself")
    
    # Prevent linking if already has a parent
    if customer.parent_customer_id is not None:
        raise HTTPException(status_code=400, detail="Customer is already linked to another account")
    
    print(f"🔗 LINKING ACCOUNTS: Customer {customer_id} ({customer.name}) → Parent {parent_id} ({parent.name})")
    
    # ⭐ NEW: Check if either account has advance payment
    customer_advance = customer.advance_payment or 0
    parent_advance = parent.advance_payment or 0
    total_advance = customer_advance + parent_advance
    
    print(f"   Customer advance: ₹{customer_advance}, Parent advance: ₹{parent_advance}, Total: ₹{total_advance}")
    
    # Link the accounts first
    customer.parent_customer_id = parent_id
    db.add(customer)
    db.commit()
    
    # ⭐ NEW: If there's any advance, use it to settle dues across both accounts
    settlement_message = None
    if total_advance > 0:
        print(f"   Found ₹{total_advance} in advance payments, settling dues...")
        
        # Get all dues from both accounts (FIFO)
        all_dues = (
            db.query(Sale)
            .filter(Sale.customer_id.in_([customer_id, parent_id]), Sale.due_amount > 0)
            .order_by(Sale.date.asc(), Sale.id.asc())
            .all()
        )
        
        print(f"   Found {len(all_dues)} sales with dues:")
        for s in all_dues:
            print(f"     - Sale {s.id}: customer_id={s.customer_id}, date={s.date}, due=₹{s.due_amount}")
        
        # Apply advance using FIFO
        remaining_advance = total_advance
        settled_count = 0
        
        for due_sale in all_dues:
            if remaining_advance <= 0:
                break
            
            if remaining_advance >= due_sale.due_amount:
                remaining_advance -= due_sale.due_amount
                due_sale.amount_paid += due_sale.due_amount
                due_sale.due_amount = 0
                settled_count += 1
                print(f"     ✅ Fully settled sale {due_sale.id}, remaining=₹{remaining_advance}")
            else:
                due_sale.amount_paid += remaining_advance
                due_sale.due_amount -= remaining_advance
                settled_count += 1
                print(f"     ⚠️ Partially settled sale {due_sale.id}, new due=₹{due_sale.due_amount}")
                remaining_advance = 0
            
            db.add(due_sale)
        
        # Clear advance from child account and store final advance in parent
        customer.advance_payment = 0
        parent.advance_payment = remaining_advance
        
        db.add(customer)
        db.add(parent)
        db.commit()
        
        print(f"   Settlement complete: {settled_count} sales settled, ₹{remaining_advance} remaining advance")
        
        if settled_count > 0:
            settlement_message = f"Used ₹{total_advance - remaining_advance:.2f} advance to settle {settled_count} sale(s). Remaining advance: ₹{remaining_advance:.2f}"
        else:
            settlement_message = f"No dues to settle. ₹{remaining_advance:.2f} stored as advance in parent account."
    
    response = {
        "message": f"Successfully linked {customer.name} to {parent.name}",
        "customer_id": customer_id,
        "parent_id": parent_id
    }
    
    if settlement_message:
        response["settlement_message"] = settlement_message
        response["advance_payment"] = parent.advance_payment
    
    return response


@router.post("/{customer_id}/unlink")
def unlink_customer_account(customer_id: int, db: Session = Depends(get_db)):
    """
    Unlink a customer account from its parent.
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    if customer.parent_customer_id is None:
        raise HTTPException(status_code=400, detail="Customer is not linked to any account")
    
    parent_id = customer.parent_customer_id
    customer.parent_customer_id = None
    db.commit()
    db.refresh(customer)
    
    return {
        "message": f"Successfully unlinked {customer.name}",
        "customer_id": customer_id,
        "previous_parent_id": parent_id
    }


@router.get("/{customer_id}/linked-accounts")
def get_linked_accounts(customer_id: int, db: Session = Depends(get_db)):
    """
    Get all linked accounts (parent + children) for a customer.
    Returns the parent and all its children.
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Determine the parent ID
    if customer.parent_customer_id:
        parent_id = customer.parent_customer_id
    else:
        parent_id = customer.id
    
    # Get parent
    parent = db.query(Customer).filter(Customer.id == parent_id).first()
    
    # Get all children
    children = db.query(Customer).filter(Customer.parent_customer_id == parent_id).all()
    
    return {
        "parent": parent,
        "children": children,
        "total_accounts": 1 + len(children)
    }


@router.get("/{customer_id}/combined-bill")
def get_combined_bill(customer_id: int, db: Session = Depends(get_db)):
    """
    Get combined bill for linked accounts.
    Includes sales, dues, and jar tracking for parent + all children.
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Determine the parent ID
    if customer.parent_customer_id:
        parent_id = customer.parent_customer_id
    else:
        parent_id = customer.id
    
    # Get all linked accounts
    parent = db.query(Customer).filter(Customer.id == parent_id).first()
    children = db.query(Customer).filter(Customer.parent_customer_id == parent_id).all()
    all_accounts = [parent] + children
    all_ids = [acc.id for acc in all_accounts]
    
    # Get combined sales with dues
    due_sales = (
        db.query(Sale)
        .filter(Sale.customer_id.in_(all_ids), Sale.due_amount > 0)
        .order_by(Sale.date.desc())
        .all()
    )
    
    # Calculate totals
    total_due = sum(s.due_amount for s in due_sales)
    
    # Get jar tracking
    total_jars_due = 0
    for acc_id in all_ids:
        jt = db.query(JarTracking).filter(JarTracking.customer_id == acc_id).first()
        if jt:
            total_jars_due += jt.current_due_jars
    
    # Get last payment
    last_payment = (
        db.query(PaymentHistory)
        .filter(PaymentHistory.customer_id.in_(all_ids))
        .order_by(PaymentHistory.date.desc())
        .first()
    )
    
    return {
        "accounts": [{"id": acc.id, "name": acc.name, "type": "parent" if acc.id == parent_id else "child"} for acc in all_accounts],
        "total_due": float(total_due),
        "total_jars_due": int(total_jars_due),
        "pending_sales": [
            {
                "id": s.id,
                "customer_id": s.customer_id,
                "customer_name": next((acc.name for acc in all_accounts if acc.id == s.customer_id), ""),
                "date": s.date,
                "total_cost": s.total_cost,
                "amount_paid": s.amount_paid,
                "due_amount": s.due_amount,
            }
            for s in due_sales
        ],
        "last_payment": {
            "amount_paid": last_payment.amount_paid,
            "date": last_payment.date,
        } if last_payment else None,
    }
