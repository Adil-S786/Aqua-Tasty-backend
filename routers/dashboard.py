# backend/routers/dashboard.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional

from dependencies import get_db
from models import Sale, Expense, Customer, JarTracking, PaymentHistory

router = APIRouter(tags=["Dashboard"])


@router.get("/summary")
def get_summary(db: Session = Depends(get_db)):
    total_income = db.query(func.coalesce(func.sum(Sale.amount_paid), 0.0)).scalar() or 0.0
    total_due = db.query(func.coalesce(func.sum(Sale.due_amount), 0.0)).scalar() or 0.0
    total_expense = db.query(func.coalesce(func.sum(Expense.amount), 0.0)).scalar() or 0.0
    total_our_jars_out = db.query(func.coalesce(func.sum(Sale.our_jars), 0)).scalar() or 0
    return {
        "total_income": float(total_income),
        "total_due": float(total_due),
        "total_expense": float(total_expense),
        "net_profit": float(total_income - total_expense),
        "total_our_jars_out": int(total_our_jars_out)
    }


@router.get("/dashboard/stats")
def get_dashboard_stats(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Dashboard filtered stats. Accepts start_date and end_date (YYYY-MM-DD)"""
    sdate = start_date
    edate = end_date

    # SALES FILTERED
    sale_q = db.query(Sale)
    if sdate:
        sale_q = sale_q.filter(func.date(Sale.date) >= sdate)
    if edate:
        sale_q = sale_q.filter(func.date(Sale.date) <= edate)
    sales = sale_q.all()

    # PAYMENTS FILTERED
    pay_q = db.query(PaymentHistory)
    if sdate:
        pay_q = pay_q.filter(func.date(PaymentHistory.date) >= sdate)
    if edate:
        pay_q = pay_q.filter(func.date(PaymentHistory.date) <= edate)
    payments = pay_q.all()

    # EXPENSE FILTERED
    exp_q = db.query(Expense)
    if sdate:
        exp_q = exp_q.filter(func.date(Expense.date) >= sdate)
    if edate:
        exp_q = exp_q.filter(func.date(Expense.date) <= edate)
    expenses = exp_q.all()

    # CUSTOMERS FILTERED
    cust_q = db.query(Customer)
    if sdate:
        cust_q = cust_q.filter(func.date(Customer.created_at) >= sdate)
    if edate:
        cust_q = cust_q.filter(func.date(Customer.created_at) <= edate)
    new_customers = cust_q.count()

    # METRICS
    total_sale = sum(s.total_cost for s in sales)
    due_amount = sum(s.due_amount for s in sales)

    sale_received = float(sum((s.amount_paid or 0.0) for s in sales))
    total_received = float(sum((p.amount_paid or 0.0) for p in payments))
    due_received = max(0.0, total_received - sale_received)

    walkin_sales = len([s for s in sales if s.customer_id is None])
    profiled_sales = len([s for s in sales if s.customer_id is not None])

    total_orders = len(sales)

    total_jars = sum(s.total_jars for s in sales)
    jar_due = sum(s.our_jars for s in sales)

    # Jar returned
    jt_q = db.query(JarTracking)
    if sdate:
        jt_q = jt_q.filter(func.date(JarTracking.last_update) >= sdate)
    if edate:
        jt_q = jt_q.filter(func.date(JarTracking.last_update) <= edate)
    jar_returned = sum(j.our_jars_returned for j in jt_q.all())

    total_expense = sum(e.amount for e in expenses)

    profit = total_received - total_expense

    return {
        "total_sale": float(total_sale),
        "sale_amount_received": float(sale_received),
        "due_amount_received": float(due_received),
        "total_received": float(total_received),
        "due": float(due_amount),
        "walkin_sales": walkin_sales,
        "profile_sales": profiled_sales,
        "total_orders": total_orders,
        "new_customers": new_customers,
        "total_jars_sold": total_jars,
        "jar_due": jar_due,
        "jar_returned": jar_returned,
        "expense": float(total_expense),
        "profit": float(profit),
    }


@router.get("/walkin/bill")
def get_walkin_bill(name: str, db: Session = Depends(get_db)):
    """Return bill details for a WALK-IN customer."""
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Walk-in name is required")

    due_sales = (
        db.query(Sale)
        .filter(
            Sale.customer_id.is_(None),
            func.lower(Sale.customer_name) == func.lower(clean_name),
            Sale.due_amount > 0,
        )
        .order_by(Sale.date.desc())
        .all()
    )

    total_due = sum(s.due_amount for s in due_sales)

    jt = (
        db.query(JarTracking)
        .filter(
            JarTracking.customer_id.is_(None),
            func.lower(JarTracking.customer_name) == func.lower(clean_name),
        )
        .first()
    )
    jar_due = jt.current_due_jars if jt else 0

    last_payment = (
        db.query(PaymentHistory)
        .filter(
            PaymentHistory.customer_id.is_(None),
            func.lower(PaymentHistory.customer_name) == func.lower(clean_name),
        )
        .order_by(PaymentHistory.date.desc())
        .first()
    )

    return {
        "name": clean_name,
        "jar_due": jar_due,
        "total_due": total_due,
        "pending_sales": [
            {
                "id": s.id,
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
        }
        if last_payment
        else None,
    }
