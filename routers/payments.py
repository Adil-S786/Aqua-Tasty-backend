# backend/routers/payments.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from dependencies import get_db
from models import PaymentHistory, Customer, Sale

router = APIRouter(prefix="/payments", tags=["Payments"])


@router.get("")
def list_payments(db: Session = Depends(get_db)):
    """List all payments with joined customer name if available."""
    results = (
        db.query(
            PaymentHistory.id,
            PaymentHistory.customer_id,
            PaymentHistory.customer_name,
            PaymentHistory.amount_paid,
            PaymentHistory.date,
            Customer.name.label("profile_name"),
        )
        .outerjoin(Customer, PaymentHistory.customer_id == Customer.id)
        .order_by(PaymentHistory.date.desc())
        .all()
    )

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


@router.delete("/{payment_id}")
def delete_payment(payment_id: int, db: Session = Depends(get_db)):
    """Delete a payment and REVERT its effect on Sale dues. Re-opens dues in reverse LIFO order (newest first)."""
    payment = db.query(PaymentHistory).filter(PaymentHistory.id == payment_id).first()
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")

    customer_id = payment.customer_id
    customer_name = payment.customer_name
    amount_to_revert = payment.amount_paid

    # Get sales in LIFO order (newest first) for payment reversal
    if customer_id:
        sales = (
            db.query(Sale)
            .filter(Sale.customer_id == customer_id)
            .order_by(Sale.date.desc())  # LIFO - newest first
            .all()
        )
    else:
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
            sale.amount_paid -= reversible
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
