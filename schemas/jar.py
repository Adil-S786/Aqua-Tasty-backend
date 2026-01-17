# backend/schemas/jar.py
from pydantic import BaseModel, Field
from typing import Optional


class JarReturn(BaseModel):
    customer_id: Optional[int] = None
    customer_name: Optional[str] = None
    returned_count: int = Field(..., gt=0)
