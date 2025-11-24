# backend/main.py
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator
from typing import Optional
from database import SessionLocal, engine, Base
from models import Customer, PaymentHistory, Reminder, Sale, JarTracking, Expense
from sqlalchemy import func
from fastapi import Query
from datetime import date
from fastapi.responses import JSONResponse
from fastapi import Request



Base.metadata.create_all(bind=engine)

app = FastAPI(title="Water Plant API")

origins = ["http://localhost:3000", "http://127.0.0.1:3000", "https://aqua-tasty.vercel.app"]
# For local development we also allow LAN IPs (e.g. http://192.168.1.6:3000)
# Use a regex to match 192.168.x.x origins on port 3000.
# allow_origin_regex = r"^http://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):3000$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- DB -----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ----------------- Schemas -----------------
class CustomerCreate(BaseModel):
    name: str = Field(..., min_length=1)
    phone: Optional[str] = None
    address: Optional[str] = None
    fixed_price_per_jar: Optional[float] = None
    delivery_type: Optional[str] = "self"

    @validator("delivery_type")
    def validate_delivery_type(cls, v):
        if v not in ("delivery", "self"):
            raise ValueError("delivery_type must be 'delivery' or 'self'")
        return v


class CustomerUpdate(BaseModel):
    name: str
    phone: Optional[str] = None
    address: Optional[str] = None
    fixed_price_per_jar: Optional[float] = None
    delivery_type: Optional[str] = "self"


class SaleCreate(BaseModel):
    is_profiled: bool = True
    customer_id: Optional[int] = None
    customer_name: Optional[str] = None
    total_jars: int = Field(..., gt=0)
    customer_own_jars: int = Field(0, ge=0)
    cost_per_jar: Optional[float] = None
    amount_paid: float = Field(0, ge=0)
    sale_date: Optional[str] = None   # "2025-01-10"


class JarReturn(BaseModel):
    customer_id: Optional[int] = None
    customer_name: Optional[str] = None
    returned_count: int = Field(..., gt=0)


class ExpenseCreate(BaseModel):
    description: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)


class ConvertWalkIn(BaseModel):
    # Payload when converting a walk-in to a profiled customer
    customer_name: str   # existing walk-in name
    name: str            # new profiled name
    phone: Optional[str] = None
    address: Optional[str] = None
    fixed_price_per_jar: Optional[float] = None
    delivery_type: Optional[str] = "self"

    @validator("delivery_type")
    def validate_delivery_type(cls, v):
        if v not in ("delivery", "self"):
            raise ValueError("delivery_type must be 'delivery' or 'self'")
        return v


class ReminderCreate(BaseModel):
    customer_id: Optional[int] = None
    custom_name: Optional[str] = None
    reason: str
    frequency: int = 0   # integer days (0 = one-time)
    next_date: datetime
    note: Optional[str] = None
    status: str = "pending"

class ReminderUpdate(BaseModel):
    reason: Optional[str] = None
    frequency: Optional[int] = None
    next_date: Optional[datetime] = None
    note: Optional[str] = None
    status: Optional[str] = None

class ReminderOut(BaseModel):
    id: int
    customer_id: Optional[int]
    custom_name: Optional[str]
    reason: str
    frequency: int
    next_date: datetime
    note: Optional[str]
    status: str
    created_at: datetime

    class Config:
        orm_mode = True
    
    @validator("next_date", pre=True)
    def format_dt(cls, v):
        return v.isoformat() if isinstance(v, datetime) else v

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


def normalize_name(name: str) -> str:
    """Normalize customer name – trim + lowercase compare + title case storage."""
    clean = (name or "").strip()
    if not clean:
        return ""
    return clean.title()      # Example: 'adIl' → 'Adil'






# ----------------- Customer endpoints -----------------
@app.post("/customers")
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
                frequency=3,           # default 3 (as you requested)
                next_date=now,
                note="Auto-created on profile creation (delivery).",
                status="pending"
            )
            db.add(default_reminder)
            db.commit()
        except Exception as e:
            # don't fail customer creation if reminder fails; log or ignore
            db.rollback()
    return {"message": "Customer created", "customer": customer}


@app.get("/customers")
def list_customers(db: Session = Depends(get_db)):
    return db.query(Customer).order_by(Customer.name).all()


@app.put("/customers/{customer_id}")
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


@app.delete("/customers/{customer_id}")
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


# ----------------- Convert Walk-in endpoint (RESTORED) -----------------
@app.post("/customers/convert-walkin")
def convert_walkin(payload: ConvertWalkIn, db: Session = Depends(get_db)):
    """
    Convert a walk-in customer (by name) into a new profiled customer.
    Behavior:
      - If a profiled customer with same name already exists -> return 400 (as requested).
      - Else -> create new Customer, reassign Sale rows and JarTracking entries for that walk-in name.
    """
    walkin_name = (payload.customer_name or "").strip()
    if not walkin_name:
        raise HTTPException(status_code=400, detail="customer_name (walk-in) required")

    new_name = (payload.name or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="name (new profile) required")

    # 1) If profiled customer with same name exists -> error (you requested this)
    existing_profile = db.query(Customer).filter(func.lower(Customer.name) == func.lower(new_name)).first()
    if existing_profile:
        raise HTTPException(status_code=400, detail="A profiled customer with this name already exists")

    # 2) Create new profiled customer
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

    # 3) Reassign sales that belong to this walk-in name (customer_id is NULL and customer_name matches)
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

    # 4) Reassign jar tracking entries for the walk-in
    #    If a JarTracking entry exists with this walkin name and no customer_id -> attach to new_customer
    jt = db.query(JarTracking).filter(JarTracking.customer_id.is_(None), func.trim(JarTracking.customer_name) == walkin_name).first()
    if jt:
        jt.customer_id = new_customer.id
        jt.customer_name = new_customer.name
        db.add(jt)
    else:
        # There might be no single jartracking row for walk-in; optionally create if sales created our_jars
        # Recalculate full jar stats
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

        jt = db.query(JarTracking).filter(
            JarTracking.customer_id == new_customer.id
        ).first()

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

    # ---------------- CHECK NAME DUPLICATION -----------------
@app.get("/customers/check-name")
def check_customer_name(name: str, db: Session = Depends(get_db)):
    clean = name.strip().lower()

    exists = (
        db.query(Customer)
        .filter(func.lower(Customer.name) == clean)
        .first()
    )

    return {"exists": bool(exists)}

# ----------------- Sales endpoints -----------------
@app.post("/sales")
def create_sale(payload: SaleCreate, db: Session = Depends(get_db)):
    # Validate own jars <= total_jars
    if payload.customer_own_jars > payload.total_jars:
        raise HTTPException(status_code=400, detail="Customer's own jars cannot exceed total jars")

    customer_id = None
    customer_name = None
    cost_per_jar = payload.cost_per_jar

    # ✅ Profiled Sale
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
                raise HTTPException(status_code=400, detail="cost_per_jar required (customer has no fixed price)")

    # ✅ Walk-in Sale — with duplicate name check and AUTO-CONVERT if profiled exists
    else:
        input_name = normalize_name(payload.customer_name or "Walk-in")

        # Check if that name matches any profiled customer (case-insensitive)
        matching_customer = (
            db.query(Customer)
            .filter(func.lower(Customer.name) == func.lower(input_name))
            .first()
        )

        if matching_customer:
            # AUTO-CONVERT: convert this sale to a profiled sale automatically
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
            # Real walk-in (no profiled match)
            customer_name = input_name
            if cost_per_jar is None:
                raise HTTPException(status_code=400, detail="cost_per_jar required for walk-in sale")

    # ✅ Compute amounts
    our_jars = payload.total_jars - payload.customer_own_jars
    total_cost = payload.total_jars * cost_per_jar
    due_amount = max(0.0, total_cost - payload.amount_paid)

    sale = Sale(
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
    db.add(sale)
    db.commit()
    db.refresh(sale)
    # ⭐ ADD PAYMENT HISTORY for amount paid during sale
    if sale.amount_paid > 0:
        payment = PaymentHistory(
            customer_id=sale.customer_id,
            customer_name=sale.customer_name,
            amount_paid=sale.amount_paid
        )
        db.add(payment)
        db.commit()

    # ✅ Update jar tracking
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
        if sale.customer_id:
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

    return sale


@app.get("/sales")
def list_sales(db: Session = Depends(get_db)):
    # Get query result as tuples
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

    # Convert each SQLAlchemy Row into a dictionary
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


@app.get("/sales/profiled")
def profiled_sales_history(db: Session = Depends(get_db)):
    return db.query(Sale).filter(Sale.customer_id != None).order_by(Sale.date.desc()).all()


@app.get("/sales/history/{customer_id}")
def sales_history_for_customer(customer_id: int, db: Session = Depends(get_db)):
    return db.query(Sale).filter(Sale.customer_id == customer_id).order_by(Sale.date.desc()).all()


@app.post("/sales/paydue")
def pay_due(
    customer_id: Optional[int] = Body(None),
    customer_name: Optional[str] = Body(None),
    amount: float = Body(..., gt=0),
    db: Session = Depends(get_db)
):
    """
    Pay due amount for either profiled or walk-in customers.
    Settles oldest sales first (FIFO order).
    """

    if not customer_id and not customer_name:
        raise HTTPException(status_code=400, detail="Customer ID or name required.")

    # Step 1: Get all due sales for this customer (sorted oldest first)
    if customer_id:
        due_sales = (
            db.query(Sale)
            .filter(Sale.customer_id == customer_id, Sale.due_amount > 0)
            .order_by(Sale.date.asc())
            .all()
        )
    else:
        due_sales = (
            db.query(Sale)
            .filter(Sale.customer_name == customer_name, Sale.due_amount > 0)
            .order_by(Sale.date.asc())
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

    # Record in payment history
    payment_record = PaymentHistory(
        customer_id=customer_id,
        customer_name=customer_name,
        amount_paid=amount - remaining,
    )
    db.add(payment_record)

    db.commit()

    # Recalculate total due for UI
    if customer_id:
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

    return {
        "message": "Due payment recorded successfully.",
        "paid_amount": amount - remaining,
        "remaining_unapplied": remaining,
        "total_due_now": total_due
    }


@app.post("/sales/total-due")
def get_total_due(
    customer_id: Optional[int] = Body(None),
    customer_name: Optional[str] = Body(None),
    db: Session = Depends(get_db)
):
    """Return total due for a profiled or walk-in customer."""
    if not customer_id and not customer_name:
        raise HTTPException(status_code=400, detail="Customer ID or name required")

    if customer_id:
        total_due = (
            db.query(func.coalesce(func.sum(Sale.due_amount), 0.0))
            .filter(Sale.customer_id == customer_id)
            .scalar()
        )
    else:
        total_due = (
            db.query(func.coalesce(func.sum(Sale.due_amount), 0.0))
            .filter(Sale.customer_name == customer_name)
            .scalar()
        )

    return {"total_due": float(total_due)}


# ----------------- Jar tracking -----------------
@app.get("/jartracking")
def get_jartracking(db: Session = Depends(get_db)):
    return db.query(JarTracking).order_by(JarTracking.current_due_jars.desc()).all()


@app.post("/jartracking/return")
def return_jars(payload: JarReturn, db: Session = Depends(get_db)):
    """
    Handle jar returns using FIFO — oldest unpaid jar sales are reduced first.
    Updates both individual Sale records and overall JarTracking summary.
    """
    if payload.returned_count <= 0:
        raise HTTPException(status_code=400, detail="Returned count must be greater than 0")

    customer_id = payload.customer_id
    customer_name = payload.customer_name

    if not customer_id and not customer_name:
        raise HTTPException(status_code=400, detail="Customer ID or name required")

    # Fetch all sales with jars still due (our_jars > 0)
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

    # Deduct jars FIFO from sales
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

    # Recalculate jar tracking totals for that customer
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


# ----------------- Expenses & Summary -----------------
@app.post("/expenses")
def create_expense(payload: ExpenseCreate, db: Session = Depends(get_db)):
    e = Expense(description=payload.description, amount=payload.amount)
    db.add(e)
    db.commit()
    db.refresh(e)
    return e

@app.get("/expenses")
def list_expenses(db: Session = Depends(get_db)):
    return db.query(Expense).order_by(Expense.date.desc()).all()


@app.get("/summary")
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


@app.get("/payments")
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


@app.get("/walkin/bill")
def get_walkin_bill(name: str, db: Session = Depends(get_db)):
    """
    Return bill details for a WALK-IN customer.
    Uses customer_name field because walk-ins have no customer_id.
    """

    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Walk-in name is required")

    # ---------------- SALES WITH DUES ----------------
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

    # ---------------- JAR TRACKING ----------------
    jt = (
        db.query(JarTracking)
        .filter(
            JarTracking.customer_id.is_(None),
            func.lower(JarTracking.customer_name) == func.lower(clean_name),
        )
        .first()
    )
    jar_due = jt.current_due_jars if jt else 0

    # ---------------- LAST PAYMENT ----------------
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

@app.put("/expenses/{expense_id}")
def update_expense(expense_id: int, data: dict, db: Session = Depends(get_db)):
    expense = db.query(Expense).filter(Expense.id == expense_id).first()
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")

    expense.description = data.get("description", expense.description)
    expense.amount = data.get("amount", expense.amount)

    db.commit()
    db.refresh(expense)

    return {"message": "Expense updated", "expense": expense}


@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: int, db: Session = Depends(get_db)):
    expense = db.query(Expense).filter(Expense.id == expense_id).first()
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")

    db.delete(expense)
    db.commit()

    return {"message": "Expense deleted"}


@app.delete("/sales/{sale_id}")
def delete_sale(sale_id: int, db: Session = Depends(get_db)):
    sale = db.query(Sale).filter(Sale.id == sale_id).first()
    if not sale:
        raise HTTPException(404, "Sale not found")

    customer_id = sale.customer_id
    customer_name = sale.customer_name

    # DELETE SALE
    db.delete(sale)
    db.commit()

    # Recalculate jartracking after deleting this sale
    recalc_jartracking(db, customer_id, customer_name)

    # Recalculate summary
    recalc_summary(db)

    return {"message": "Sale deleted successfully and jar due updated."}



@app.put("/sales/{sale_id}")
def update_sale(sale_id: int, payload: SaleCreate, db: Session = Depends(get_db)):
    """
    Correct sale update logic:
    - Remove the old sale effect completely.
    - Insert a new updated sale record (same logic as create_sale).
    - Recalculate jar tracking from scratch.
    - Recalculate summary at the end.
    """

    # ----- 1) Get old sale -----
    old = db.query(Sale).filter(Sale.id == sale_id).first()
    if not old:
        raise HTTPException(status_code=404, detail="Sale not found")

    old_customer_id = old.customer_id
    old_customer_name = old.customer_name

    # ----- 2) Delete old sale -----
    db.delete(old)
    db.commit()

    # ----- 3) Reuse CREATE_SALE LOGIC -----
    # (because update must behave same as new sale creation)
    customer_id = None
    customer_name = None
    cost_per_jar = payload.cost_per_jar

    # Profiled customer
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
        # Walk-in update
        input_name = normalize_name(payload.customer_name or "Walk-in")

        # Auto convert if matching profiled exists
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

    # ----- 4) Compute new sale fields -----
    our_jars = payload.total_jars - payload.customer_own_jars
    total_cost = payload.total_jars * cost_per_jar
    due_amount = max(0.0, total_cost - payload.amount_paid)

    # ----- 5) Create NEW updated sale -----
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
        date=payload.sale_date or None,   # NEW
    )
    db.add(new_sale)
    db.commit()
    db.refresh(new_sale)

    # ----- 6) RECALCULATE JarTracking completely -----
    def recalc_jars(cid=None, cname=None):
        if cid:
            sales = db.query(Sale).filter(Sale.customer_id == cid).all()
        else:
            sales = db.query(Sale).filter(Sale.customer_name == cname).all()

        total_given = sum((s.total_jars - s.customer_own_jars) for s in sales)
        total_remaining = sum(s.our_jars for s in sales)

        # Find JarTracking
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

    # Recalc for updated new sale
    recalc_jars(customer_id, customer_name)

    # Also recalc for old customer if changed
    if (old_customer_id != customer_id) or (old_customer_name != customer_name):
        recalc_jars(old_customer_id, old_customer_name)

    # ----- 7) Summary auto-updates for frontend -----
    # (front-end calls /summary after refreshAll)

    return {"message": "Sale updated successfully", "sale": new_sale}

@app.get("/dashboard/stats")
def get_dashboard_stats(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Dashboard filtered stats.
    Accepts start_date and end_date (YYYY-MM-DD)
    """

    # ---------------- RANGE ----------------
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

    # ---------------- METRICS ----------------
    total_sale = sum(s.total_cost for s in sales)
    # sale_received = sum(s.amount_paid for s in sales)
    due_amount = sum(s.due_amount for s in sales)

    # payments made (due money received)
    # due_received = sum(p.amount_paid for p in payments)

    # ----------- TODAY'S DATE -----------
    today = func.current_date()

    # ----------- SALE AMOUNT RECEIVED (SALE DATE = TODAY) -----------
    sale_received = float(sum((s.amount_paid or 0.0) for s in sales))

    total_received = float(sum((p.amount_paid or 0.0) for p in payments))

    # ----------- DUE AMOUNT RECEIVED (SALE DATE < TODAY) -----------
    # due_received = payments in range that are NOT part of sales in the same range
    # (guard against negative due_received if something odd happens)
    due_received = max(0.0, total_received - sale_received)



    walkin_sales = len([s for s in sales if s.customer_id is None])
    profiled_sales = len([s for s in sales if s.customer_id is not None])

    total_orders = len(sales)

    total_jars = sum(s.total_jars for s in sales)
    jar_due = sum(s.our_jars for s in sales)

    # Jar returned = sum of our_jars_returned in JarTracking within range
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


    # ---------------- Reminders ----------------

@app.post("/reminders", response_model=ReminderOut)
def create_reminder(payload: ReminderCreate, db: Session = Depends(get_db)):

    if not payload.customer_id and not payload.custom_name:
        raise HTTPException(400, "Either customer_id or custom_name is required")

    r = Reminder(
        customer_id=payload.customer_id,
        custom_name=payload.custom_name,
        reason=payload.reason,
        frequency=payload.frequency,
        next_date=payload.next_date,
        note=payload.note,
        status=payload.status,
    )

    db.add(r)
    db.commit()
    db.refresh(r)

    return r


@app.get("/reminders")
def list_reminders(db: Session = Depends(get_db)):
    reminders = (
        db.query(Reminder)
        .order_by(Reminder.next_date.asc())
        .all()
    )

    profiled = []
    customs = []

    for r in reminders:
        name = None
        if r.customer_id:
            cust = db.query(Customer).filter(Customer.id == r.customer_id).first()
            name = cust.name if cust else None

        item = {
            "id": r.id,
            "customer_id": r.customer_id,
            "customer_name": name,
            "custom_name": r.custom_name,
            "reason": r.reason,
            "frequency": r.frequency,
            "next_date": r.next_date,
            "note": r.note,
            "status": r.status,
            "created_at": r.created_at,
        }

        if r.customer_id:
            profiled.append(item)
        else:
            customs.append(item)

    return {"profiled": profiled, "customs": customs}



@app.get("/reminders/{reminder_id}", response_model=ReminderOut)
def get_reminder(reminder_id: int, db: Session = Depends(get_db)):
    r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not r:
        raise HTTPException(404, "Reminder not found")
    return r


@app.put("/reminders/{reminder_id}", response_model=ReminderOut)
def update_reminder(reminder_id: int, payload: ReminderUpdate, db: Session = Depends(get_db)):

    r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not r:
        raise HTTPException(404, "Reminder not found")

    for key, value in payload.dict(exclude_unset=True).items():
        setattr(r, key, value)

    db.commit()
    db.refresh(r)
    return r


@app.delete("/reminders/{reminder_id}")
def delete_reminder(reminder_id: int, db: Session = Depends(get_db)):

    r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not r:
        raise HTTPException(404, "Reminder not found")

    db.delete(r)
    db.commit()

    return {"message": "Reminder deleted successfully"}


@app.post("/reminders/{reminder_id}/status")
def update_reminder_status(
    reminder_id: int,
    status: str = Body(...),
    db: Session = Depends(get_db)
):
    r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not r:
        raise HTTPException(404, "Reminder not found")

    if status == "skipped" and r.frequency and r.frequency > 0:
        # move to next frequency days
        r.next_date = r.next_date + timedelta(days=r.frequency)
        r.status = "scheduled"
    else:
        r.status = status
    db.commit()
    db.refresh(r)
    return r


@app.get("/reminders/due/today", response_model=list[ReminderOut])
def get_today_reminders(db: Session = Depends(get_db)):
    today = func.date(func.now())
    return (
        db.query(Reminder)
        .filter(func.date(Reminder.next_date) == today)
        .filter(Reminder.status == "pending")
        .order_by(Reminder.next_date.asc())
        .all()
    )


@app.get("/reminders/overdue", response_model=list[ReminderOut])
def get_overdue_reminders(db: Session = Depends(get_db)):
    now = func.now()
    return (
        db.query(Reminder)
        .filter(Reminder.next_date < now)
        .filter(Reminder.status == "pending")
        .all()
    )


@app.post("/reminders/{reminder_id}/advance")
def advance_next_date(reminder_id: int, db: Session = Depends(get_db)):
    r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not r:
        raise HTTPException(404, "Reminder not found")

    freq = r.frequency or 0
    if freq == 0:
        # one-time reminder — mark completed or leave
        raise HTTPException(400, "Cannot advance a one-time reminder (frequency = 0).")
    # add freq days
    r.next_date = r.next_date + timedelta(days=freq)
    r.status = "scheduled"
    db.commit()
    db.refresh(r)
    return r

@app.post("/reminders/auto-reschedule-daily")
def auto_reschedule_daily(db: Session = Depends(get_db)):
    today = datetime.now()
    rows = (
        db.query(Reminder)
        .filter(Reminder.status == "pending")
        .all()
    )
    for r in rows:
        r.next_date = today + timedelta(days=1)
        r.status = "scheduled"
        db.add(r)

    db.commit()
    return {"updated": len(rows)}


@app.delete("/reminders/cleanup")
def cleanup_old_reminders(db: Session = Depends(get_db)):
    week_ago = datetime.now() - timedelta(days=7)
    removed = (
        db.query(Reminder)
        .filter(Reminder.next_date < week_ago)
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"deleted": removed}
