# backend/services/smart_reminder_service.py
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import Customer, Sale, Reminder
from services.activity_service import detect_activity_status  # ⭐ Import the function

# ⭐ IST Timezone Helper (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now():
    """Get current datetime in IST"""
    return datetime.now(IST)


def analyze_customer_pattern(customer_id: int, db: Session):
    """Analyze customer's purchase pattern and return average days between purchases"""
    sales = (
        db.query(Sale)
        .filter(Sale.customer_id == customer_id)
        .order_by(Sale.date.desc())
        .limit(10)
        .all()
    )

    if len(sales) < 2:
        return None  # Not enough data

    # Calculate average days between purchases
    intervals = []
    for i in range(len(sales) - 1):
        days = (sales[i].date - sales[i + 1].date).days
        if days > 0:  # Only count positive intervals
            intervals.append(days)

    if not intervals:
        return None

    avg_days = sum(intervals) / len(intervals)
    return round(avg_days)


def generate_smart_reminders(db: Session, max_inactive_days: int = 60):
    """
    Generate reminders for ACTIVE DELIVERY customers who are due based on their patterns.
    Only creates reminders for customers with:
    - activity_status = 'active'
    - delivery_type = 'delivery' (not self-pickup)
    
    Args:
        max_inactive_days: Maximum days since last sale to consider customer active (default: 60)
    """
    # ⭐ NEW: Only get ACTIVE DELIVERY customers
    customers = db.query(Customer).filter(
        Customer.active == True,
        Customer.activity_status == "active",
        Customer.delivery_type == "delivery"  # ⭐ Only delivery customers
    ).all()
    
    created = 0
    skipped = 0
    inactive = 0
    no_pattern = 0

    for customer in customers:
        # Check if already has pending reminder
        existing = (
            db.query(Reminder)
            .filter(
                Reminder.customer_id == customer.id,
                Reminder.status.in_(["pending", "scheduled"])
            )
            .first()
        )

        if existing:
            skipped += 1
            continue  # Skip if already has reminder

        # Get last sale
        last_sale = (
            db.query(Sale)
            .filter(Sale.customer_id == customer.id)
            .order_by(Sale.date.desc())
            .first()
        )

        if not last_sale:
            no_pattern += 1
            continue

        # Calculate days since last purchase (using IST)
        now_ist = get_ist_now()
        sale_date = last_sale.date
        
        # Convert sale_date to IST for comparison
        if sale_date.tzinfo is not None:
            sale_date_ist = sale_date.astimezone(IST)
        else:
            # Assume UTC if no timezone
            sale_date_ist = sale_date.replace(tzinfo=timezone.utc).astimezone(IST)
        
        days_since_last = (now_ist - sale_date_ist).days

        # ⭐ NEW: Skip if customer inactive for too long (default: 60 days)
        if days_since_last > max_inactive_days:
            inactive += 1
            continue  # Customer likely stopped buying

        # Analyze pattern
        avg_days = analyze_customer_pattern(customer.id, db)
        if not avg_days or avg_days < 1:
            no_pattern += 1
            continue

        # ⭐ NEW: Only create reminder if within reasonable range
        # Customer is "due" if: days_since_last >= (avg_days - 1)
        # But not if they're way overdue (more than 2x their normal frequency)
        max_overdue = avg_days * 2
        
        if days_since_last >= (avg_days - 1) and days_since_last <= max_overdue:
            # Calculate expected next delivery date in IST
            expected_date = sale_date_ist + timedelta(days=avg_days)
            # If expected date is in the past, use now
            if expected_date < now_ist:
                expected_date = now_ist
            
            reminder = Reminder(
                customer_id=customer.id,
                reason="delivery",
                frequency=avg_days,
                next_date=expected_date,
                note=f"Auto-generated: Customer typically orders every {avg_days} days. Last order was {days_since_last} days ago.",
                status="pending",
            )
            db.add(reminder)
            created += 1
        elif days_since_last > max_overdue:
            # Customer is way overdue - might have stopped buying
            inactive += 1
        else:
            # Not due yet
            skipped += 1

    db.commit()
    return {
        "created": created,
        "skipped": skipped,
        "inactive": inactive,
        "no_pattern": no_pattern,
        "total_customers": len(customers),
        "message": f"Created {created} reminders. {inactive} customers appear inactive (>{max_inactive_days} days since last order)."
    }


def update_customer_reminder_after_sale(customer_id: int, db: Session):
    """
    Update or create reminder after a sale is made.
    
    ⭐ NEW BEHAVIOR:
    - If customer was inactive, making a purchase will reactivate them
    - Activity status is recalculated based on new purchase
    - Only creates reminders for ACTIVE DELIVERY customers
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        return
    
    # ⭐ NEW: Recalculate activity status after purchase (this may change inactive → active)
    new_status = detect_activity_status(customer_id, db)
    customer.activity_status = new_status
    db.commit()
    
    # Only proceed with reminder creation if customer is now active and has delivery
    if customer.activity_status != "active" or customer.delivery_type != "delivery":
        return
    
    # Find pending reminders for this customer
    pending = (
        db.query(Reminder)
        .filter(
            Reminder.customer_id == customer_id,
            Reminder.status.in_(["pending", "scheduled"])
        )
        .all()
    )

    # Mark them as completed since customer just made a purchase
    for reminder in pending:
        reminder.status = "completed"

    # Analyze pattern and create new reminder
    avg_days = analyze_customer_pattern(customer_id, db)
    if avg_days and avg_days > 0:
        # Use IST for new reminder
        now_ist = get_ist_now()
        next_date = now_ist + timedelta(days=avg_days)
        
        new_reminder = Reminder(
            customer_id=customer_id,
            reason="delivery",
            frequency=avg_days,
            next_date=next_date,
            note=f"Auto-created after sale on {now_ist.strftime('%Y-%m-%d')}",
            status="scheduled",
        )
        db.add(new_reminder)

    db.commit()


def auto_advance_overdue_reminders(db: Session, days_overdue: int = 1):
    """
    Auto-advance overdue reminders - ONLY for ACTIVE DELIVERY customers
    - Yesterday's reminders → Move to TODAY
    - Older reminders → Skip cycles to next occurrence
    - Only for customers with delivery_type = 'delivery'
    
    Args:
        days_overdue: How many days overdue before auto-advancing (default: 1)
    
    Returns:
        Number of reminders advanced
    """
    now_ist = get_ist_now()
    today_ist = now_ist.date()
    yesterday_ist = today_ist - timedelta(days=1)
    cutoff_date_ist = now_ist - timedelta(days=days_overdue)
    
    # Find overdue reminders with frequency > 0, ONLY for ACTIVE DELIVERY customers
    overdue_reminders = (
        db.query(Reminder)
        .join(Customer, Reminder.customer_id == Customer.id)
        .filter(
            Reminder.next_date < cutoff_date,
            Reminder.status.in_(["pending", "scheduled"]),
            Reminder.frequency > 0,
            Customer.activity_status == "active",  # Only active customers
            Customer.delivery_type == "delivery"  # ⭐ NEW: Only delivery customers
        )
        .all()
    )
    
    advanced = 0
    moved_to_today = 0
    
    for reminder in overdue_reminders:
        # Convert to IST for comparison
        next_date = reminder.next_date
        if next_date.tzinfo is not None:
            next_date_ist = next_date.astimezone(IST)
        else:
            next_date_ist = next_date.replace(tzinfo=timezone.utc).astimezone(IST)
        
        days_overdue_count = (now_ist - next_date_ist).days
        reminder_date_ist = next_date_ist.date()
        
        # ⭐ NEW: If reminder was yesterday, just move to today
        if reminder_date_ist == yesterday_ist:
            # Move to today in IST (same time)
            new_datetime_ist = datetime.combine(today_ist, next_date_ist.time())
            reminder.next_date = new_datetime_ist.replace(tzinfo=IST)
            
            reminder.status = "pending"
            
            # Update note
            if reminder.note:
                reminder.note += f" | Moved from yesterday to today on {now_ist.strftime('%Y-%m-%d')}"
            else:
                reminder.note = f"Moved from yesterday to today on {now_ist.strftime('%Y-%m-%d')}"
            
            moved_to_today += 1
        else:
            # For older reminders, calculate cycles to skip
            cycles_to_skip = (days_overdue_count // reminder.frequency) + 1
            
            # Advance to next occurrence
            reminder.next_date = next_date_ist + timedelta(days=reminder.frequency * cycles_to_skip)
            reminder.status = "scheduled"
            
            # Update note
            if reminder.note:
                reminder.note += f" | Auto-advanced {cycles_to_skip} cycle(s) on {now_ist.strftime('%Y-%m-%d')}"
            else:
                reminder.note = f"Auto-advanced {cycles_to_skip} cycle(s) on {now_ist.strftime('%Y-%m-%d')}"
        
        advanced += 1
    
    db.commit()
    return {
        "total_advanced": advanced,
        "moved_to_today": moved_to_today,
        "skipped_cycles": advanced - moved_to_today
    }
