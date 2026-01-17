# backend/schemas/reminder.py
from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime


class ReminderCreate(BaseModel):
    customer_id: Optional[int] = None
    custom_name: Optional[str] = None
    reason: str
    frequency: int = 0  # integer days (0 = one-time)
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
