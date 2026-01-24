# backend/routers/sales.py
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
from datetime import date

from dependencies import get_db
from schemas import SaleCreate
from models import Customer, Sale, JarTracking, PaymentHistory, Reminder
from utils import normalize_name
from services import recalc_jartracking
from services.activity_service import update_customer_activity_status

router = APIRouter(prefix="/sales", tags=["Sales"])


@router.post("")
def create_sale(payload: SaleCreate, db: Session = Depends(get_db)):
    if payload.customer_own_jars > payload.total_jars:
        raise HTTPException(status_code=400, detail="Customer's own jars cannot exceed total jars")

    customer_id = None
    customer_name = None
    cost_per_jar = payload.cost_per_jar
    customer = None

    # Profiled Sale
    if payload.is_profiled:
        if not payload.customer_id:
            raise HTTPException(status_code=400, detail="Please select a profiled customer")

        customer = db.query(Customer).filter(Customer.id == payload.customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")

        customer_id = customer.id
        customer_name = customer.name

        if cost_per_jar is None:
            if customer.fixed_price_per_jar is not None:
                cost_per_jar = customer.fixed_price_per_jar
            else:
                raise HTTPException(status_code=400, detail="cost_per_jar required (customer has no fixed price)")

    # Walk-in Sale with AUTO-CONVERT
    else:
        input_name = normalize_name(payload.customer_name or "Walk-in")

        matching_customer = (
            db.query(Customer)
            .filter(func.lower(Customer.name) == func.lower(input_name))
            .first()
        )

        if matching_customer:
            customer = matching_customer
            customer_id = matching_customer.id
            customer_name = matching_customer.name
            payload.is_profiled = True
            if cost_per_jar is None:
                cost_per_jar = (
                    matching_customer.fixed_price_per_jar
                    if matching_customer.fixed_price_per_jar is not None
                    else payload.cost_per_jar
                )
        else:
            customer_name = input_name
            if cost_per_jar is None:
                raise HTTPException(status_code=400, detail="cost_per_jar required for walk-in sale")

    our_jars = payload.total_jars - payload.customer_own_jars
    total_cost = payload.total_jars * cost_per_jar
    
    # ⭐ PROPER ADVANCE + PAYMENT LOGIC
    advance_payment_message = None
    
    # Calculate total payment needed (old dues + current sale)
    total_payment_needed = total_cost
    
    if customer_id:
        # Get all old dues (FIFO - oldest first)
        old_dues = (
            db.query(Sale)
            .filter(Sale.customer_id == customer_id, Sale.due_amount > 0)
            .order_by(Sale.date.asc())
            .all()
        )
        
        total_old_dues = sum(s.due_amount for s in old_dues)
        total_payment_needed += total_old_dues
        
        # Total available = advance + actual payment
        existing_advance = customer.advance_payment if customer else 0
        total_available = existing_advance + payload.amount_paid
        
        # Settle old dues first (FIFO)
        remaining_payment = total_available
        for old_sale in old_dues:
            if remaining_payment <= 0:
                break
            
            if remaining_payment >= old_sale.due_amount:
                remaining_payment -= old_sale.due_amount
                old_sale.amount_paid += old_sale.due_amount
                old_sale.due_amount = 0
            else:
                old_sale.amount_paid += remaining_payment
                old_sale.due_amount -= remaining_payment
                remaining_payment = 0
            
            db.add(old_sale)
        
        # After settling old dues, use remaining for current sale
        amount_paid_for_current = min(remaining_payment, total_cost)
        due_amount = total_cost - amount_paid_for_current
        
        # Calculate final advance
        final_advance = max(0, remaining_payment - total_cost)
        
        # Update customer advance and show message
        if customer:
            advance_change = final_advance - existing_advance
            customer.advance_payment = final_advance
            
            if existing_advance > 0 and final_advance < existing_advance:
                advance_payment_message = f"₹{existing_advance - final_advance:.2f} advance used. Remaining: ₹{final_advance:.2f}"
            elif final_advance > existing_advance:
                advance_payment_message = f"₹{advance_change:.2f} added to advance. Total advance: ₹{final_advance:.2f}"
            elif final_advance > 0 and existing_advance == 0:
                advance_payment_message = f"₹{final_advance:.2f} recorded as advance payment"
            
            db.add(customer)
    else:
        # Walk-in customer - no advance payment
        amount_paid_for_current = payload.amount_paid
        due_amount = max(0, total_cost - amount_paid_for_current)

    sale = Sale(
        customer_id=customer_id,
        customer_name=customer_name,
        total_jars=payload.total_jars,
        customer_own_jars=payload.customer_own_jars,
        our_jars=our_jars,
        cost_per_jar=cost_per_jar,
        total_cost=total_cost,
        amount_paid=amount_paid_for_current,
        due_amount=due_amount,
        date=payload.sale_date or None,
    )
    db.add(sale)
    db.commit()
    db.refresh(sale)

    if payload.amount_paid > 0:
        payment = PaymentHistory(
            customer_id=sale.customer_id,
            customer_name=sale.customer_name,
            amount_paid=payload.amount_paid
        )
        db.add(payment)
        db.commit()

    if our_jars > 0:
        jt = None
        if customer_id:
            jt = db.query(JarTracking).filter(JarTracking.customer_id == customer_id).first()
        else:
            jt = db.query(JarTracking).filter(JarTracking.customer_name == customer_name).first()

        if jt:
            jt.our_jars_given = (jt.our_jars_given or 0) + our_jars
            jt.current_due_jars = max(0, (jt.our_jars_given or 0) - (jt.our_jars_returned or 0))
        else:
            jt = JarTracking(
                customer_id=customer_id,
                customer_name=customer_name,
                our_jars_given=our_jars,
                our_jars_returned=0,
                current_due_jars=our_jars,
            )
            db.add(jt)

        db.commit()
        db.refresh(jt)

    # ⭐ ENHANCED: Update reminders after sale (moved outside jar block)
    if sale.customer_id:
        # Mark today's reminder as completed
        today = date.today()
        todays_reminder = (
            db.query(Reminder)
            .filter(
                Reminder.customer_id == sale.customer_id,
                func.date(Reminder.next_date) == today,
                Reminder.status.in_(["pending", "scheduled", "rescheduled"])
            )
            .first()
        )

        if todays_reminder:
            todays_reminder.status = "completed"
            db.add(todays_reminder)
            db.commit()
        
        # ⭐ Auto-update customer reminders and activity status based on pattern
        from services.smart_reminder_service import update_customer_reminder_after_sale
        update_customer_reminder_after_sale(sale.customer_id, db)

    response = {
        "sale": sale,
        "message": "Sale created successfully"
    }
    
    if advance_payment_message:
        response["advance_payment_message"] = advance_payment_message
    
    return response


@router.get("")
def list_sales(db: Session = Depends(get_db)):
    results = (
        db.query(
            Sale.id,
            Sale.customer_id,
            Sale.customer_name,
            Customer.name.label("profile_name"),
            Sale.total_jars,
            Sale.customer_own_jars,
            Sale.our_jars,
            Sale.total_cost,
            Sale.amount_paid,
            Sale.due_amount,
            Sale.date
        )
        .outerjoin(Customer, Sale.customer_id == Customer.id)
        .order_by(Sale.date.desc())
        .all()
    )

    sales_list = [
        {
            "id": r.id,
            "customer_id": r.customer_id,
            "customer_name": r.customer_name,
            "profile_name": r.profile_name,
            "total_jars": r.total_jars,
            "customer_own_jars": r.customer_own_jars,
            "our_jars": r.our_jars,
            "total_cost": r.total_cost,
            "amount_paid": r.amount_paid,
            "due_amount": r.due_amount,
            "date": r.date,
        }
        for r in results
    ]

    return sales_list


@router.get("/profiled")
def profiled_sales_history(db: Session = Depends(get_db)):
    return db.query(Sale).filter(Sale.customer_id != None).order_by(Sale.date.desc()).all()


@router.get("/history/{customer_id}")
def sales_history_for_customer(customer_id: int, db: Session = Depends(get_db)):
    return db.query(Sale).filter(Sale.customer_id == customer_id).order_by(Sale.date.desc()).all()


@router.post("/paydue")
def pay_due(
    customer_id: Optional[int] = Body(None),
    customer_name: Optional[str] = Body(None),
    amount: float = Body(..., gt=0),
    db: Session = Depends(get_db)
):
    """Pay due amount for either profiled or walk-in customers. Settles oldest sales first (FIFO).
    For linked accounts, settles dues across all linked accounts."""
    if not customer_id and not customer_name:
        raise HTTPException(status_code=400, detail="Customer ID or name required.")

    # Get customer if profiled
    customer = None
    account_ids = []
    
    if customer_id:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Determine all account IDs to include (for linked accounts)
        account_ids = [customer_id]
        
        if customer.parent_customer_id:
            # This is a child, get parent and all siblings
            parent_id = customer.parent_customer_id
            account_ids = [parent_id]
            children = db.query(Customer).filter(Customer.parent_customer_id == parent_id).all()
            account_ids.extend([c.id for c in children])
        else:
            # Check if this is a parent with children
            children = db.query(Customer).filter(Customer.parent_customer_id == customer_id).all()
            if children:
                account_ids.extend([c.id for c in children])
        
        # Get all due sales from all linked accounts
        due_sales = (
            db.query(Sale)
            .filter(Sale.customer_id.in_(account_ids), Sale.due_amount > 0)
            .order_by(Sale.date.asc())  # FIFO - oldest first across all accounts
            .all()
        )
    else:
        due_sales = (
            db.query(Sale)
            .filter(Sale.customer_name == customer_name, Sale.due_amount > 0)
            .order_by(Sale.date.asc())  # FIFO - oldest first
            .all()
        )

    if not due_sales:
        # No dues, treat as advance payment
        if customer:
            customer.advance_payment = (customer.advance_payment or 0.0) + amount
            db.add(customer)
            
            payment_record = PaymentHistory(
                customer_id=customer_id,
                customer_name=customer_name,
                amount_paid=amount,
            )
            db.add(payment_record)
            db.commit()
            
            return {
                "message": "No dues found. Amount recorded as advance payment.",
                "paid_amount": amount,
                "advance_payment": customer.advance_payment,
                "total_due_now": 0
            }
        else:
            raise HTTPException(status_code=404, detail="No due sales found for this customer.")

    remaining = amount
    settled_accounts = set()  # Track which accounts had dues settled
    
    for sale in due_sales:
        if remaining <= 0:
            break

        if remaining >= sale.due_amount:
            remaining -= sale.due_amount
            sale.amount_paid += sale.due_amount
            sale.due_amount = 0
        else:
            sale.amount_paid += remaining
            sale.due_amount -= remaining
            remaining = 0

        db.add(sale)
        settled_accounts.add(sale.customer_id)

    # If there's remaining amount after settling all dues, record as advance
    advance_message = None
    if remaining > 0 and customer:
        customer.advance_payment = (customer.advance_payment or 0.0) + remaining
        db.add(customer)
        advance_message = f"₹{remaining:.2f} recorded as advance payment"

    # Create payment records for all accounts that had dues settled
    if customer_id and len(account_ids) > 1:
        # For linked accounts, create payment record for the account being paid through
        payment_record = PaymentHistory(
            customer_id=customer_id,
            customer_name=customer_name,
            amount_paid=amount,
        )
        db.add(payment_record)
    else:
        payment_record = PaymentHistory(
            customer_id=customer_id,
            customer_name=customer_name,
            amount_paid=amount,
        )
        db.add(payment_record)

    db.commit()

    # Calculate total remaining due across all linked accounts
    if customer_id and len(account_ids) > 1:
        total_due = (
            db.query(func.sum(Sale.due_amount))
            .filter(Sale.customer_id.in_(account_ids))
            .scalar()
        ) or 0.0
    elif customer_id:
        total_due = (
            db.query(func.sum(Sale.due_amount))
            .filter(Sale.customer_id == customer_id)
            .scalar()
        ) or 0.0
    else:
        total_due = (
            db.query(func.sum(Sale.due_amount))
            .filter(Sale.customer_name == customer_name)
            .scalar()
        ) or 0.0

    response = {
        "message": "Due payment recorded successfully.",
        "paid_amount": amount - remaining,
        "total_due_now": total_due,
        "settled_accounts": len(settled_accounts) if customer_id else 1
    }
    
    if advance_message:
        response["advance_payment_message"] = advance_message
        response["advance_payment"] = customer.advance_payment if customer else 0
    
    return response


@router.post("/total-due")
def get_total_due(
    customer_id: Optional[int] = Body(None),
    customer_name: Optional[str] = Body(None),
    db: Session = Depends(get_db)
):
    """Return total due for a profiled or walk-in customer. For linked accounts, returns combined total."""
    if not customer_id and not customer_name:
        raise HTTPException(status_code=400, detail="Customer ID or name required")

    if customer_id:
        # Check if customer has linked accounts
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        print(f"DEBUG: Customer {customer_id} ({customer.name}), parent_customer_id: {customer.parent_customer_id}")
        
        # Determine all account IDs to include
        account_ids = [customer_id]
        
        if customer.parent_customer_id:
            # This is a child, get parent and all siblings
            parent_id = customer.parent_customer_id
            print(f"DEBUG: This is a child, parent_id: {parent_id}")
            account_ids = [parent_id]
            children = db.query(Customer).filter(Customer.parent_customer_id == parent_id).all()
            account_ids.extend([c.id for c in children])
            print(f"DEBUG: Found {len(children)} children: {[c.id for c in children]}")
        else:
            # Check if this is a parent with children
            children = db.query(Customer).filter(Customer.parent_customer_id == customer_id).all()
            if children:
                account_ids.extend([c.id for c in children])
                print(f"DEBUG: This is a parent with {len(children)} children: {[c.id for c in children]}")
            else:
                print(f"DEBUG: This is a standalone account (no parent, no children)")
        
        print(f"DEBUG: Final account_ids to query: {account_ids}")
        
        # Get total due from all linked accounts
        total_due = (
            db.query(func.coalesce(func.sum(Sale.due_amount), 0.0))
            .filter(Sale.customer_id.in_(account_ids))
            .scalar()
        )
        
        # Get individual dues for debugging
        for acc_id in account_ids:
            individual_due = (
                db.query(func.coalesce(func.sum(Sale.due_amount), 0.0))
                .filter(Sale.customer_id == acc_id)
                .scalar()
            )
            acc = db.query(Customer).filter(Customer.id == acc_id).first()
            print(f"DEBUG: Account {acc_id} ({acc.name if acc else 'Unknown'}): ₹{individual_due}")
        
        print(f"DEBUG: Total due calculated: ₹{total_due}")
    else:
        total_due = (
            db.query(func.coalesce(func.sum(Sale.due_amount), 0.0))
            .filter(Sale.customer_name == customer_name)
            .scalar()
        )

    return {"total_due": float(total_due), "account_ids": account_ids if customer_id else []}  # Return account_ids for debugging


@router.delete("/{sale_id}")
def delete_sale(sale_id: int, db: Session = Depends(get_db)):
    sale = db.query(Sale).filter(Sale.id == sale_id).first()
    if not sale:
        raise HTTPException(404, "Sale not found")

    customer_id = sale.customer_id
    customer_name = sale.customer_name

    db.delete(sale)
    db.commit()

    recalc_jartracking(db, customer_id, customer_name)

    return {"message": "Sale deleted successfully and jar due updated."}


@router.put("/{sale_id}")
def update_sale(sale_id: int, payload: SaleCreate, db: Session = Depends(get_db)):
    """Update sale by deleting old and creating new with updated values."""
    old = db.query(Sale).filter(Sale.id == sale_id).first()
    if not old:
        raise HTTPException(status_code=404, detail="Sale not found")

    old_customer_id = old.customer_id
    old_customer_name = old.customer_name

    db.delete(old)
    db.commit()

    customer_id = None
    customer_name = None
    cost_per_jar = payload.cost_per_jar

    if payload.is_profiled:
        if not payload.customer_id:
            raise HTTPException(status_code=400, detail="Please select a profiled customer")

        cust = db.query(Customer).filter(Customer.id == payload.customer_id).first()
        if not cust:
            raise HTTPException(status_code=404, detail="Customer not found")

        customer_id = cust.id
        customer_name = cust.name

        if cost_per_jar is None:
            if cust.fixed_price_per_jar is not None:
                cost_per_jar = cust.fixed_price_per_jar
            else:
                raise HTTPException(status_code=400, detail="cost_per_jar required for this customer")
    else:
        input_name = normalize_name(payload.customer_name or "Walk-in")
        matching_customer = db.query(Customer).filter(func.lower(Customer.name) == func.lower(input_name)).first()
        if matching_customer:
            customer_id = matching_customer.id
            customer_name = matching_customer.name
            payload.is_profiled = True
            if cost_per_jar is None:
                cost_per_jar = matching_customer.fixed_price_per_jar
        else:
            customer_name = input_name
            if cost_per_jar is None:
                raise HTTPException(status_code=400, detail="cost_per_jar required for walk-in update")

    our_jars = payload.total_jars - payload.customer_own_jars
    total_cost = payload.total_jars * cost_per_jar
    due_amount = max(0.0, total_cost - payload.amount_paid)

    new_sale = Sale(
        customer_id=customer_id,
        customer_name=customer_name,
        total_jars=payload.total_jars,
        customer_own_jars=payload.customer_own_jars,
        our_jars=our_jars,
        cost_per_jar=cost_per_jar,
        total_cost=total_cost,
        amount_paid=payload.amount_paid,
        due_amount=due_amount,
        date=payload.sale_date or None,
    )
    db.add(new_sale)
    db.commit()
    db.refresh(new_sale)

    def recalc_jars(cid=None, cname=None):
        if cid:
            sales = db.query(Sale).filter(Sale.customer_id == cid).all()
        else:
            sales = db.query(Sale).filter(Sale.customer_name == cname).all()

        total_given = sum((s.total_jars - s.customer_own_jars) for s in sales)
        total_remaining = sum(s.our_jars for s in sales)

        if cid:
            jt = db.query(JarTracking).filter(JarTracking.customer_id == cid).first()
        else:
            jt = db.query(JarTracking).filter(JarTracking.customer_name == cname).first()

        if jt:
            jt.our_jars_given = total_given
            jt.current_due_jars = total_remaining
        else:
            jt = JarTracking(
                customer_id=cid,
                customer_name=cname,
                our_jars_given=total_given,
                our_jars_returned=0,
                current_due_jars=total_remaining,
            )
        db.add(jt)
        db.commit()

    recalc_jars(customer_id, customer_name)

    if (old_customer_id != customer_id) or (old_customer_name != customer_name):
        recalc_jars(old_customer_id, old_customer_name)

    return {"message": "Sale updated successfully", "sale": new_sale}
