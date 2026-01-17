# backend/routers/jars.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from dependencies import get_db
from schemas import JarReturn
from models import Sale, JarTracking

router = APIRouter(prefix="/jartracking", tags=["Jar Tracking"])


@router.get("")
def get_jartracking(db: Session = Depends(get_db)):
    return db.query(JarTracking).order_by(JarTracking.current_due_jars.desc()).all()


@router.post("/return")
def return_jars(payload: JarReturn, db: Session = Depends(get_db)):
    """Handle jar returns using FIFO — oldest unpaid jar sales are reduced first."""
    if payload.returned_count <= 0:
        raise HTTPException(status_code=400, detail="Returned count must be greater than 0")

    customer_id = payload.customer_id
    customer_name = payload.customer_name

    if not customer_id and not customer_name:
        raise HTTPException(status_code=400, detail="Customer ID or name required")

    if customer_id:
        sales = (
            db.query(Sale)
            .filter(Sale.customer_id == customer_id, Sale.our_jars > 0)
            .order_by(Sale.date.asc())
            .all()
        )
    else:
        sales = (
            db.query(Sale)
            .filter(Sale.customer_name == customer_name, Sale.our_jars > 0)
            .order_by(Sale.date.asc())
            .all()
        )

    if not sales:
        raise HTTPException(status_code=404, detail="No jar dues found for this customer")

    remaining = payload.returned_count

    for sale in sales:
        if remaining <= 0:
            break

        if sale.our_jars <= remaining:
            remaining -= sale.our_jars
            sale.our_jars = 0
        else:
            sale.our_jars -= remaining
            remaining = 0

        db.add(sale)

    db.commit()

    total_given = (
        db.query(func.sum(Sale.total_jars - Sale.customer_own_jars))
        .filter(
            Sale.customer_id == customer_id if customer_id else Sale.customer_name == customer_name
        )
        .scalar()
        or 0
    )

    total_remaining = (
        db.query(func.sum(Sale.our_jars))
        .filter(
            Sale.customer_id == customer_id if customer_id else Sale.customer_name == customer_name
        )
        .scalar()
        or 0
    )

    returned_now = payload.returned_count - remaining

    jt = None
    if customer_id:
        jt = db.query(JarTracking).filter(JarTracking.customer_id == customer_id).first()
    elif customer_name:
        jt = db.query(JarTracking).filter(JarTracking.customer_name == customer_name).first()

    if jt:
        jt.our_jars_returned = (jt.our_jars_returned or 0) + returned_now
        jt.current_due_jars = max(0, total_remaining)
    else:
        jt = JarTracking(
            customer_id=customer_id,
            customer_name=customer_name,
            our_jars_given=total_given,
            our_jars_returned=returned_now,
            current_due_jars=total_remaining,
        )
        db.add(jt)

    db.commit()
    db.refresh(jt)

    return {
        "message": f"Returned {returned_now} jars (FIFO applied). Remaining jars due: {jt.current_due_jars}",
        "remaining_due": jt.current_due_jars,
        "total_returned_now": returned_now,
    }
