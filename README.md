# Water Plant API - Backend

FastAPI backend for water plant management system with customer tracking, sales, expenses, jar inventory, and smart reminders.

## Directory Structure

```
backend/
├── main.py                 # FastAPI app initialization
├── database.py             # Database connection and session management
├── models.py               # SQLAlchemy ORM models
├── dependencies.py         # Shared dependencies (e.g., get_db)
│
├── schemas/                # Pydantic models for request/response validation
│   ├── customer.py         # Customer-related schemas
│   ├── sale.py             # Sale-related schemas
│   ├── expense.py          # Expense schemas
│   ├── jar.py              # Jar tracking schemas
│   └── reminder.py         # Reminder schemas
│
├── routers/                # API endpoint routers by domain
│   ├── customers.py        # Customer CRUD endpoints
│   ├── sales.py            # Sales management endpoints
│   ├── expenses.py         # Expense tracking endpoints
│   ├── jars.py             # Jar tracking endpoints
│   ├── payments.py         # Payment history endpoints
│   ├── reminders.py        # Reminder management with smart suggestions
│   └── dashboard.py        # Dashboard stats and summary endpoints
│
├── services/               # Business logic layer
│   ├── jar_service.py      # Jar tracking recalculation logic
│   ├── summary_service.py  # Summary calculation logic
│   └── smart_reminder_service.py  # Smart reminder pattern detection
│
└── utils/                  # Helper functions
    └── helpers.py          # Utility functions (e.g., normalize_name)
```

## Features

- **Customer Management**: Track customers with profiles, delivery types, and custom pricing
- **Sales Tracking**: Record sales with jar tracking, payment management, and due amounts
- **Expense Management**: Track business expenses with date filtering
- **Jar Inventory**: Automatic jar tracking per customer with return management
- **Payment History**: Complete payment records with reversal capability
- **Smart Reminders**: AI-powered reminder system that learns customer patterns
  - Analyzes purchase history to suggest optimal reminder frequencies
  - Auto-generates reminders for all customers based on patterns
  - Auto-advances overdue reminders
  - Handles inactive customers intelligently
- **Dashboard Analytics**: Real-time stats with date filtering and summaries

## Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL database

### Local Development

1. **Clone the repository**
   ```bash
   cd backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   Create a `.env` file:
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/waterplant
   ```

4. **Run the server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Production Deployment

The app is configured for Heroku deployment via `Procfile`:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Database tables are created automatically on startup.
