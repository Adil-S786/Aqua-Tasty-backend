# backend/routers/payments.py
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
from datetime import datetime, timezone, timedelta

from dependencies import get_db
from models import PaymentHistory, Customer, Sale
from routers.sales import apply_payment_fifo  # ⭐ Import shared function

router = APIRouter(prefix="/payments", tags=["Payments"])

# ⭐ IST Timezone Helper (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now():
    """Get current datetime in IST"""
    return datetime.now(IST)


@router.get("")
def list_payments(customer_id: int = None, db: Session = Depends(get_db)):
    """List all payments with joined customer name if available. Optionally filter by customer_id."""
    query = (
        db.query(
            PaymentHistory.id,
            PaymentHistory.customer_id,
            PaymentHistory.customer_name,
            PaymentHistory.amount_paid,
            PaymentHistory.date,
            Customer.name.label("profile_name"),
        )
        .outerjoin(Customer, PaymentHistory.customer_id == Customer.id)
    )
    
    # Filter by customer_id if provided
    if customer_id:
        query = query.filter(PaymentHistory.customer_id == customer_id)
    
    results = query.order_by(PaymentHistory.date.desc()).all()

    return [
        {
            "id": r.id,
            "customer_id": r.customer_id,
            "customer_name": r.customer_name or r.profile_name,
            "amount_paid": r.amount_paid,
            "date": r.date,
        }
        for r in results
    ]


@router.post("/create")
def create_backdated_payment(
    customer_id: Optional[int] = Body(None),
    customer_name: Optional[str] = Body(None),
    amount: float = Body(..., gt=0),
    payment_date: Optional[str] = Body(None),
    db: Session = Depends(get_db)
):
    """
    Create a backdated payment record and settle dues using FIFO.
    This creates a payment history entry with a custom date AND settles oldest dues first.
    """
    if not customer_id and not customer_name:
        raise HTTPException(status_code=400, detail="Customer ID or name required")
    
    # Parse payment date or use current time in IST
    if payment_date:
        try:
            parsed_date = datetime.fromisoformat(payment_date.replace('Z', '+00:00'))
        except:
            raise HTTPException(status_code=400, detail="Invalid date format")
    else:
        parsed_date = get_ist_now()
    
    # Get customer if profiled
    customer = None
    
    if customer_id:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # ⭐ Use shared FIFO payment function
        final_advance, advance_message, settled_count = apply_payment_fifo(
            customer_id=customer_id,
            amount=amount,
            db=db,
            include_advance=False  # Don't include existing advance for backdated payments
        )
        
        # Calculate total remaining due
        account_ids = [customer_id]
        if customer.parent_customer_id:
            parent_id = customer.parent_customer_id
            account_ids = [parent_id]
            children = db.query(Customer).filter(Customer.parent_customer_id == parent_id).all()
            account_ids.extend([c.id for c in children])
        else:
            children = db.query(Customer).filter(Customer.parent_customer_id == customer_id).all()
            if children:
                account_ids.extend([c.id for c in children])
        
        total_due = (
            db.query(func.sum(Sale.due_amount))
            .filter(Sale.customer_id.in_(account_ids))
            .scalar()
        ) or 0.0
    else:
        # Walk-in customer
        due_sales = (
            db.query(Sale)
            .filter(Sale.customer_name == customer_name, Sale.due_amount > 0)
            .order_by(Sale.date.asc())  # FIFO - oldest first
            .all()
        )
        
        # Settle dues using FIFO
        remaining = amount
        settled_count = 0
        
        for sale in due_sales:
            if remaining <= 0:
                break
            
            if remaining >= sale.due_amount:
                remaining -= sale.due_amount
                sale.amount_paid += sale.due_amount
                sale.due_amount = 0
                settled_count += 1
            else:
                sale.amount_paid += remaining
                sale.due_amount -= remaining
                settled_count += 1
                remaining = 0
            
            db.add(sale)
        
        total_due = (
            db.query(func.sum(Sale.due_amount))
            .filter(Sale.customer_name == customer_name)
            .scalar()
        ) or 0.0
        
        advance_message = None
    
    # Create payment record with custom date
    payment = PaymentHistory(
        customer_id=customer_id,
        customer_name=customer_name,
        amount_paid=amount,
        date=parsed_date
    )
    
    db.add(payment)
    db.commit()
    db.refresh(payment)
    
    response = {
        "message": "Backdated payment recorded and dues settled successfully",
        "payment": {
            "id": payment.id,
            "customer_id": payment.customer_id,
            "customer_name": payment.customer_name,
            "amount_paid": payment.amount_paid,
            "date": payment.date
        },
        "settled_sales": settled_count,
        "total_due_now": total_due
    }
    
    if advance_message:
        response["advance_payment_message"] = advance_message
        response["advance_payment"] = final_advance
    
    return response


@router.delete("/{payment_id}")
def delete_payment(payment_id: int, db: Session = Depends(get_db)):
    """
    Delete a payment and REVERT its effect.
    
    Logic:
    1. First reduce advance from parent account (if linked) or current account
    2. If advance goes negative, re-open dues in LIFO order (newest first)
    3. For linked accounts, re-opens dues across ALL linked accounts
    """
    payment = db.query(PaymentHistory).filter(PaymentHistory.id == payment_id).first()
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")

    customer_id = payment.customer_id
    customer_name = payment.customer_name
    amount_to_revert = payment.amount_paid
    
    print(f"🗑️ DELETE PAYMENT: ID={payment_id}, Amount=₹{amount_to_revert}, Customer={customer_id or customer_name}")

    if customer_id:
        # Get customer and determine linked accounts
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        
        # Find the account that holds advance (parent if linked)
        advance_customer = customer
        account_ids = [customer_id]
        
        if customer.parent_customer_id:
            # This is a child, get parent
            parent = db.query(Customer).filter(Customer.id == customer.parent_customer_id).first()
            if parent:
                advance_customer = parent
                account_ids = [parent.id]
                # Get all siblings
                children = db.query(Customer).filter(Customer.parent_customer_id == parent.id).all()
                account_ids.extend([c.id for c in children])
        else:
            # Check if this is a parent with children
            children = db.query(Customer).filter(Customer.parent_customer_id == customer_id).all()
            if children:
                account_ids.extend([c.id for c in children])
        
        print(f"   Advance customer: {advance_customer.id} ({advance_customer.name})")
        print(f"   All linked account_ids: {account_ids}")
        
        # Step 1: Reduce advance first
        current_advance = advance_customer.advance_payment or 0
        remaining_to_revert = amount_to_revert
        
        if current_advance > 0:
            if current_advance >= remaining_to_revert:
                # Advance covers the full revert
                advance_customer.advance_payment = current_advance - remaining_to_revert
                remaining_to_revert = 0
                print(f"   Reduced advance by ₹{amount_to_revert}, new advance=₹{advance_customer.advance_payment}")
            else:
                # Use all advance, still need to revert more
                remaining_to_revert -= current_advance
                advance_customer.advance_payment = 0
                print(f"   Used all advance (₹{current_advance}), still need to revert ₹{remaining_to_revert}")
            
            db.add(advance_customer)
        
        # Step 2: Re-open dues in LIFO order (newest first) across all linked accounts
        if remaining_to_revert > 0:
            sales = (
                db.query(Sale)
                .filter(Sale.customer_id.in_(account_ids))
                .order_by(Sale.date.desc(), Sale.id.desc())  # LIFO - newest first
                .all()
            )
            
            print(f"   Found {len(sales)} sales to potentially re-open dues")
            
            for sale in sales:
                if remaining_to_revert <= 0:
                    break

                reversible = sale.amount_paid

                if reversible <= 0:
                    continue

                if remaining_to_revert >= reversible:
                    sale.amount_paid = 0
                    sale.due_amount += reversible
                    remaining_to_revert -= reversible
                    print(f"   Re-opened ₹{reversible} due on sale {sale.id}")
                else:
                    sale.amount_paid -= remaining_to_revert
                    sale.due_amount += remaining_to_revert
                    print(f"   Re-opened ₹{remaining_to_revert} due on sale {sale.id}")
                    remaining_to_revert = 0

                db.add(sale)
        
        db.delete(payment)
        db.commit()
        
        return {
            "message": "Payment deleted and dues restored successfully",
            "reverted_amount": amount_to_revert,
            "advance_reduced": min(current_advance, amount_to_revert),
            "dues_reopened": amount_to_revert - min(current_advance, amount_to_revert)
        }
    
    else:
        # Walk-in customer - simple reversal (no advance, no linked accounts)
        sales = (
            db.query(Sale)
            .filter(Sale.customer_name == customer_name)
            .order_by(Sale.date.desc())  # LIFO - newest first
            .all()
        )

        remaining = amount_to_revert

        for sale in sales:
            if remaining <= 0:
                break

            reversible = sale.amount_paid

            if reversible <= 0:
                continue

            if remaining >= reversible:
                sale.amount_paid = 0
                sale.due_amount += reversible
                remaining -= reversible
            else:
                sale.amount_paid -= remaining
                sale.due_amount += remaining
                remaining = 0

            db.add(sale)

        db.delete(payment)
        db.commit()

        return {
            "message": "Payment deleted and dues restored successfully",
            "reverted_amount": amount_to_revert - remaining,
            "unapplied_amount": remaining
        }
