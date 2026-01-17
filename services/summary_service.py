# backend/services/summary_service.py
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import Sale, Expense


def recalc_summary(db: Session):
    total_income = db.query(func.coalesce(func.sum(Sale.amount_paid), 0.0)).scalar() or 0.0
    total_due = db.query(func.coalesce(func.sum(Sale.due_amount), 0.0)).scalar() or 0.0
    total_expense = db.query(func.coalesce(func.sum(Expense.amount), 0.0)).scalar() or 0.0
    total_our_jars_out = db.query(func.coalesce(func.sum(Sale.our_jars), 0)).scalar() or 0

    return {
        "total_income": total_income,
        "total_due": total_due,
        "total_expense": total_expense,
        "net_profit": total_income - total_expense,
        "total_our_jars_out": total_our_jars_out,
    }
