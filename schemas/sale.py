# backend/schemas/sale.py
from pydantic import BaseModel, Field
from typing import Optional


class SaleCreate(BaseModel):
    is_profiled: bool = True
    customer_id: Optional[int] = None
    customer_name: Optional[str] = None
    total_jars: int = Field(..., gt=0)
    customer_own_jars: int = Field(0, ge=0)
    cost_per_jar: Optional[float] = None
    amount_paid: float = Field(0, ge=0)
    sale_date: Optional[str] = None  # "2025-01-10"
