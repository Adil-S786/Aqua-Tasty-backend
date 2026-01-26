# backend/routers/sales.py
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, Tuple
from datetime import date

from dependencies import get_db
from schemas import SaleCreate
from models import Customer, Sale, JarTracking, PaymentHistory, Reminder
from utils import normalize_name
from services import recalc_jartracking
from services.activity_service import update_customer_activity_status

router = APIRouter(prefix="/sales", tags=["Sales"])


# ⭐ SHARED FUNCTION: Apply payment using FIFO across linked accounts
def apply_payment_fifo(
    customer_id: int,
    amount: float,
    db: Session,
    include_advance: bool = True
) -> Tuple[float, str, int]:
    """
    Apply payment using FIFO across all linked accounts.
    
    Args:
        customer_id: The customer making the payment
        amount: The payment amount
        db: Database session
        include_advance: Whether to include existing advance in total available
    
    Returns:
        Tuple of (final_advance, advance_message, settled_count)
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    print(f"🔍 APPLY PAYMENT FIFO: Customer {customer_id} ({customer.name}), Amount: ₹{amount}")
    
    # ⭐ Get existing advance from PARENT account (if linked) or current account
    advance_customer = customer
    if customer.parent_customer_id:
        parent = db.query(Customer).filter(Customer.id == customer.parent_customer_id).first()
        if parent:
            advance_customer = parent
            print(f"   Getting advance from PARENT account {parent.id} ({parent.name})")
    
    existing_advance = advance_customer.advance_payment or 0 if include_advance else 0
    total_available = existing_advance + amount
    
    print(f"   Existing advance: ₹{existing_advance}, Total available: ₹{total_available}")
    
    # Determine all linked account IDs
    account_ids = [customer_id]
    
    if customer.parent_customer_id:
        # This is a child, get parent and all siblings
        parent_id = customer.parent_customer_id
        print(f"   This is a CHILD account, parent_id: {parent_id}")
        account_ids = [parent_id]
        children = db.query(Customer).filter(Customer.parent_customer_id == parent_id).all()
        account_ids.extend([c.id for c in children])
        print(f"   All linked account_ids: {account_ids}")
    else:
        # Check if this is a parent with children
        children = db.query(Customer).filter(Customer.parent_customer_id == customer_id).all()
        if children:
            account_ids.extend([c.id for c in children])
            print(f"   This is a PARENT account with {len(children)} children")
            print(f"   All linked account_ids: {account_ids}")
        else:
            print(f"   This is a STANDALONE account")
    
    # Get ALL dues from all linked accounts (FIFO - oldest first)
    all_dues = (
        db.query(Sale)
        .filter(Sale.customer_id.in_(account_ids), Sale.due_amount > 0)
        .order_by(Sale.date.asc(), Sale.id.asc())  # FIFO - oldest first
        .all()
    )
    
    print(f"   Found {len(all_dues)} sales with dues across all linked accounts:")
    for s in all_dues:
        print(f"     - Sale {s.id}: customer_id={s.customer_id}, date={s.date}, due=₹{s.due_amount}")
    
    # Apply payment using FIFO
    remaining_payment = total_available
    settled_count = 0
    
    for due_sale in all_dues:
        if remaining_payment <= 0:
            break
        
        if remaining_payment >= due_sale.due_amount:
            # Fully settle this sale
            remaining_payment -= due_sale.due_amount
            due_sale.amount_paid += due_sale.due_amount
            due_sale.due_amount = 0
            settled_count += 1
            print(f"     ✅ Fully settled sale {due_sale.id}, remaining=₹{remaining_payment}")
        else:
            # Partially settle this sale
            due_sale.amount_paid += remaining_payment
            due_sale.due_amount -= remaining_payment
            settled_count += 1
            print(f"     ⚠️ Partially settled sale {due_sale.id}, new due=₹{due_sale.due_amount}")
            remaining_payment = 0
        
        db.add(due_sale)
    
    # Calculate final advance
    final_advance = max(0, remaining_payment)
    
    print(f"   Final advance: ₹{final_advance} (was ₹{existing_advance})")
    
    # Update advance in the appropriate account (parent if linked, self if standalone)
    advance_customer.advance_payment = final_advance
    db.add(advance_customer)
    
    # Generate advance payment message
    advance_message = None
    if existing_advance > 0 and final_advance < existing_advance:
        advance_message = f"₹{existing_advance - final_advance:.2f} advance used. Remaining: ₹{final_advance:.2f}"
    elif final_advance > existing_advance:
        advance_message = f"₹{final_advance - existing_advance:.2f} added to advance. Total advance: ₹{final_advance:.2f}"
    elif final_advance > 0 and existing_advance == 0:
        advance_message = f"₹{final_advance:.2f} recorded as advance payment"
    
    return final_advance, advance_message, settled_count


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
    
    # ⭐ NEW LOGIC: Create sale first with full due amount, then apply payment via FIFO
    # Step 1: Create the sale with full due amount (not paid yet)
    sale = Sale(
        customer_id=customer_id,
        customer_name=customer_name,
        total_jars=payload.total_jars,
        customer_own_jars=payload.customer_own_jars,
        our_jars=our_jars,
        cost_per_jar=cost_per_jar,
        total_cost=total_cost,
        amount_paid=0,  # Start with 0, will be updated by FIFO
        due_amount=total_cost,  # Full amount is due initially
        date=payload.sale_date or None,
    )
    db.add(sale)
    db.commit()
    db.refresh(sale)
    
    # Step 2: Apply payment using FIFO across all linked accounts (if any payment was made)
    advance_payment_message = None
    
    if payload.amount_paid > 0 and customer_id:
        # Use shared FIFO payment function
        final_advance, advance_payment_message, settled_count = apply_payment_fifo(
            customer_id=customer_id,
            amount=payload.amount_paid,
            db=db,
            include_advance=True
        )
        
        # Record payment in payment_history
        payment = PaymentHistory(
            customer_id=customer_id,
            customer_name=customer_name,
            amount_paid=payload.amount_paid
        )
        db.add(payment)
        
        db.commit()
        db.refresh(sale)  # Refresh to get updated amount_paid and due_amount
    elif payload.amount_paid > 0:
        # Walk-in customer - simple payment (no linked accounts, no advance)
        if payload.amount_paid >= total_cost:
            sale.amount_paid = total_cost
            sale.due_amount = 0
        else:
            sale.amount_paid = payload.amount_paid
            sale.due_amount = total_cost - payload.amount_paid
        
        db.add(sale)
        
        # Record payment
        payment = PaymentHistory(
            customer_id=None,
            customer_name=customer_name,
            amount_paid=payload.amount_paid
        )
        db.add(payment)
        db.commit()
        db.refresh(sale)

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

    # Handle profiled customers
    if customer_id:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Use shared FIFO payment function
        final_advance, advance_message, settled_count = apply_payment_fifo(
            customer_id=customer_id,
            amount=amount,
            db=db,
            include_advance=False  # Don't include existing advance for "Pay Due" button
        )
        
        # Create payment record
        payment_record = PaymentHistory(
            customer_id=customer_id,
            customer_name=customer_name,
            amount_paid=amount,
        )
        db.add(payment_record)
        db.commit()
        
        # Calculate total remaining due across all linked accounts
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
        
        response = {
            "message": "Due payment recorded successfully.",
            "paid_amount": amount,
            "total_due_now": total_due,
            "settled_accounts": len(set([s.customer_id for s in db.query(Sale.customer_id).filter(Sale.customer_id.in_(account_ids)).distinct()]))
        }
        
        if advance_message:
            response["advance_payment_message"] = advance_message
            response["advance_payment"] = final_advance
        
        return response
    
    # Handle walk-in customers
    else:
        due_sales = (
            db.query(Sale)
            .filter(Sale.customer_name == customer_name, Sale.due_amount > 0)
            .order_by(Sale.date.asc())  # FIFO - oldest first
            .all()
        )

        if not due_sales:
            raise HTTPException(status_code=404, detail="No due sales found for this customer.")

        remaining = amount
        
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

        payment_record = PaymentHistory(
            customer_id=None,
            customer_name=customer_name,
            amount_paid=amount,
        )
        db.add(payment_record)
        db.commit()

        total_due = (
            db.query(func.sum(Sale.due_amount))
            .filter(Sale.customer_name == customer_name)
            .scalar()
        ) or 0.0

        return {
            "message": "Due payment recorded successfully.",
            "paid_amount": amount,
            "total_due_now": total_due,
            "settled_accounts": 1
        }


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
