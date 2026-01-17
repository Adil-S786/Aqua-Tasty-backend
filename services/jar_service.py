# backend/services/jar_service.py
from sqlalchemy.orm import Session
from models import Sale, JarTracking


def recalc_jartracking(db: Session, customer_id=None, customer_name=None):
    """Recalculate exact jar due for a customer (profiled OR walk-in)."""

    # fetch all sales
    if customer_id:
        sales = db.query(Sale).filter(Sale.customer_id == customer_id).all()
    else:
        sales = db.query(Sale).filter(Sale.customer_name == customer_name).all()

    total_given = sum((s.total_jars - s.customer_own_jars) for s in sales)
    total_remaining = sum(s.our_jars for s in sales)

    # get or create jartracking record
    if customer_id:
        jt = db.query(JarTracking).filter(JarTracking.customer_id == customer_id).first()
    else:
        jt = db.query(JarTracking).filter(JarTracking.customer_name == customer_name).first()

    if jt:
        jt.our_jars_given = total_given
        jt.current_due_jars = total_remaining
        db.add(jt)
    else:
        jt = JarTracking(
            customer_id=customer_id,
            customer_name=customer_name,
            our_jars_given=total_given,
            our_jars_returned=total_given - total_remaining,
            current_due_jars=total_remaining,
        )
        db.add(jt)

    db.commit()
