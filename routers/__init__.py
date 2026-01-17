# backend/routers/__init__.py
from .customers import router as customers_router
from .sales import router as sales_router
from .expenses import router as expenses_router
from .jars import router as jars_router
from .payments import router as payments_router
from .reminders import router as reminders_router
from .dashboard import router as dashboard_router

__all__ = [
    "customers_router",
    "sales_router",
    "expenses_router",
    "jars_router",
    "payments_router",
    "reminders_router",
    "dashboard_router",
]
