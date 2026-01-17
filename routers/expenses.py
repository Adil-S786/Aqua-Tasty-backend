# backend/routers/expenses.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from dependencies import get_db
from schemas import ExpenseCreate
from models import Expense

router = APIRouter(prefix="/expenses", tags=["Expenses"])


@router.post("")
def create_expense(payload: ExpenseCreate, db: Session = Depends(get_db)):
    e = Expense(description=payload.description, amount=payload.amount)
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


@router.get("")
def list_expenses(db: Session = Depends(get_db)):
    return db.query(Expense).order_by(Expense.date.desc()).all()


@router.put("/{expense_id}")
def update_expense(expense_id: int, data: dict, db: Session = Depends(get_db)):
    expense = db.query(Expense).filter(Expense.id == expense_id).first()
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")

    expense.description = data.get("description", expense.description)
    expense.amount = data.get("amount", expense.amount)

    db.commit()
    db.refresh(expense)

    return {"message": "Expense updated", "expense": expense}


@router.delete("/{expense_id}")
def delete_expense(expense_id: int, db: Session = Depends(get_db)):
    expense = db.query(Expense).filter(Expense.id == expense_id).first()
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")

    db.delete(expense)
    db.commit()

    return {"message": "Expense deleted"}
