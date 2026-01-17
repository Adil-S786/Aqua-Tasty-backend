# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import Base, engine
from routers import (
    customers_router,
    sales_router,
    expenses_router,
    jars_router,
    payments_router,
    reminders_router,
    dashboard_router,
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(title="Water Plant API")

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://aqua-tasty.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(customers_router)
app.include_router(sales_router)
app.include_router(expenses_router)
app.include_router(jars_router)
app.include_router(payments_router)
app.include_router(reminders_router)
app.include_router(dashboard_router)


@app.get("/")
def root():
    return {"message": "Water Plant API - Refactored Structure"}
