# backend/models.py
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from database import Base

class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    phone = Column(String, nullable=True, unique=True)
    address = Column(String, nullable=True)
    fixed_price_per_jar = Column(Float, nullable=True)
    delivery_type = Column(String, default="self")  # NEW FIELD ("self" or "delivery")
    active = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())   # for new customers filtering

    sales = relationship("Sale", back_populates="customer")
    jar_entries = relationship("JarTracking", back_populates="customer")


class Sale(Base):
    __tablename__ = "sales"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True)  # null => random
    customer_name = Column(String, nullable=True)  # for random customers
    total_jars = Column(Integer, nullable=False)
    customer_own_jars = Column(Integer, default=0)
    our_jars = Column(Integer, nullable=False)  # total_jars - customer_own_jars
    cost_per_jar = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False)
    amount_paid = Column(Float, nullable=False)
    due_amount = Column(Float, nullable=False)
    date = Column(DateTime(timezone=True), server_default=func.now())

    customer = relationship("Customer", back_populates="sales")


class JarTracking(Base):
    __tablename__ = "jar_tracking"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True)
    customer_name = Column(String, nullable=True)
    our_jars_given = Column(Integer, default=0)
    our_jars_returned = Column(Integer, default=0)
    current_due_jars = Column(Integer, default=0)
    last_update = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    customer = relationship("Customer", back_populates="jar_entries")


class Expense(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True, index=True)
    description = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    date = Column(DateTime(timezone=True), server_default=func.now())

class PaymentHistory(Base):
    __tablename__ = "payment_history"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True)
    customer_name = Column(String, nullable=True)
    amount_paid = Column(Float, nullable=False)
    date = Column(DateTime(timezone=True), server_default=func.now())

class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True, index=True)

    # --- Customer Options ---
    # If profiled customer → store customer_id
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True)

    # If not profiled → store a manual name
    custom_name = Column(String, nullable=True)

    # --- Core Reminder Info ---
    reason = Column(String, nullable=False)  # delivery / due / manual / jar_return / other
    frequency = Column(Integer, default=0)  # 0..20
    # Next time to remind
    next_date = Column(DateTime(timezone=True), nullable=False)

    # Extra note (optional)
    note = Column(String, nullable=True)

    # NEW: Status (pending, completed, skipped, cancelled)
    status = Column(String, default="pending")  

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    customer = relationship("Customer")