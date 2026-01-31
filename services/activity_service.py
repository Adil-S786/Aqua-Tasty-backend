"""
Service to detect and update customer activity status based on purchase patterns.

Status Types:
- inactive: No purchase in 45+ days (auto-detected) OR manually marked
- onetime: Bought only once
- occasional: Bought 2 times only
- was_regular: Was buying regularly but hasn't bought in 22-45 days
- active: Buying regularly (purchased within last 21 days)
- no_pattern: Cannot determine pattern yet
"""

from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from models import Customer, Sale

# ⭐ IST Timezone Helper (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now():
    """Get current datetime in IST"""
    return datetime.now(IST)


def detect_activity_status(customer_id: int, db: Session) -> str:
    """
    Detect customer activity status based on purchase history.
    
    Returns one of: inactive, onetime, occasional, was_regular, active, no_pattern
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        return "no_pattern"
    
    # Get all sales for this customer, ordered by date
    sales = db.query(Sale).filter(Sale.customer_id == customer_id).order_by(Sale.date.desc()).all()
    
    if not sales:
        return "no_pattern"
    
    sale_count = len(sales)
    now_ist = get_ist_now()
    
    # Get the most recent sale date and convert to IST
    last_sale_date = sales[0].date
    if last_sale_date.tzinfo is not None:
        last_sale_date_ist = last_sale_date.astimezone(IST)
    else:
        # Assume UTC if no timezone
        last_sale_date_ist = last_sale_date.replace(tzinfo=timezone.utc).astimezone(IST)
    
    days_since_last_purchase = (now_ist - last_sale_date_ist).days
    
    # Rule 1: Only one purchase
    if sale_count == 1:
        return "onetime"
    
    # Rule 2: Exactly 2 purchases
    if sale_count == 2:
        return "occasional"
    
    # Rule 3+: 3 or more purchases - check regularity
    if sale_count >= 3:
        # Calculate average days between purchases
        sale_dates = [s.date for s in sales]
        
        # Ensure all dates are timezone-aware and convert to IST
        sale_dates = [
            d.astimezone(IST) if d.tzinfo is not None else d.replace(tzinfo=timezone.utc).astimezone(IST)
            for d in sale_dates
        ]
        
        # Calculate intervals between consecutive purchases
        intervals = []
        for i in range(len(sale_dates) - 1):
            interval = (sale_dates[i] - sale_dates[i + 1]).days
            intervals.append(interval)
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            
            # Check for inactive (45+ days since last purchase)
            if days_since_last_purchase > 45:
                return "inactive"
            
            # If average interval is <= 21 days (3 weeks), they were regular
            if avg_interval <= 21:
                # Check if they're still active
                if days_since_last_purchase <= 21:
                    return "active"
                else:
                    return "was_regular"
            else:
                # Irregular pattern but multiple purchases
                if days_since_last_purchase <= 21:
                    return "active"
                else:
                    return "occasional"
    
    return "no_pattern"


def update_customer_activity_status(customer_id: int, db: Session) -> str:
    """
    Update a customer's activity status in the database.
    Returns the new status.
    """
    status = detect_activity_status(customer_id, db)
    
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if customer:
        customer.activity_status = status
        db.commit()
        db.refresh(customer)
    
    return status


def update_all_customers_activity_status(db: Session) -> dict:
    """
    Update activity status for all customers.
    Returns a summary of status counts.
    """
    customers = db.query(Customer).all()
    
    status_counts = {
        "inactive": 0,
        "onetime": 0,
        "occasional": 0,
        "was_regular": 0,
        "active": 0,
        "no_pattern": 0
    }
    
    for customer in customers:
        status = update_customer_activity_status(customer.id, db)
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return status_counts


def mark_customer_inactive(customer_id: int, db: Session) -> bool:
    """
    Manually mark a customer as inactive and delete all pending reminders.
    Returns True if successful.
    """
    from models import Reminder
    
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if customer:
        customer.activity_status = "inactive"
        
        # ⭐ NEW: Delete all pending/scheduled reminders for this customer
        db.query(Reminder).filter(
            Reminder.customer_id == customer_id,
            Reminder.status.in_(["pending", "scheduled"])
        ).delete(synchronize_session=False)
        
        db.commit()
        db.refresh(customer)
        return True
    return False
