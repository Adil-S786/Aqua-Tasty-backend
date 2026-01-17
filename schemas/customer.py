# backend/schemas/customer.py
from pydantic import BaseModel, Field, validator
from typing import Optional


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


class ConvertWalkIn(BaseModel):
    customer_name: str  # existing walk-in name
    name: str  # new profiled name
    phone: Optional[str] = None
    address: Optional[str] = None
    fixed_price_per_jar: Optional[float] = None
    delivery_type: Optional[str] = "self"

    @validator("delivery_type")
    def validate_delivery_type(cls, v):
        if v not in ("delivery", "self"):
            raise ValueError("delivery_type must be 'delivery' or 'self'")
        return v
