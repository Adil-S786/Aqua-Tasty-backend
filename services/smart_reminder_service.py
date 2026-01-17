# backend/services/smart_reminder_service.py
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import Customer, Sale, Reminder


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
    Generate reminders for customers who are due based on their patterns
    
    Args:
        max_inactive_days: Maximum days since last sale to consider customer active (default: 60)
    """
    customers = db.query(Customer).filter(Customer.active == True).all()
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

        # Calculate days since last purchase
        days_since_last = (datetime.now() - last_sale.date).days

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
            # Calculate expected next delivery date
            expected_date = last_sale.date + timedelta(days=avg_days)
            
            # If expected date is in the past, use today
            if expected_date < datetime.now():
                expected_date = datetime.now()
            
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
    """Update or create reminder after a sale is made"""
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
        # ⭐ FIXED: Calculate from TODAY (actual purchase date), not old reminder date
        next_date = datetime.now() + timedelta(days=avg_days)
        new_reminder = Reminder(
            customer_id=customer_id,
            reason="delivery",
            frequency=avg_days,
            next_date=next_date,
            note=f"Auto-created after sale: Next delivery expected in {avg_days} days (from {datetime.now().strftime('%Y-%m-%d')})",
            status="scheduled",
        )
        db.add(new_reminder)

    db.commit()


def auto_advance_overdue_reminders(db: Session, days_overdue: int = 1):
    """
    Auto-advance overdue reminders to next occurrence
    
    Args:
        days_overdue: How many days overdue before auto-advancing (default: 1)
    
    Returns:
        Number of reminders advanced
    """
    cutoff_date = datetime.now() - timedelta(days=days_overdue)
    
    # Find overdue reminders with frequency > 0
    overdue_reminders = (
        db.query(Reminder)
        .filter(
            Reminder.next_date < cutoff_date,
            Reminder.status.in_(["pending", "scheduled"]),
            Reminder.frequency > 0
        )
        .all()
    )
    
    advanced = 0
    for reminder in overdue_reminders:
        # Calculate how many cycles to skip
        days_overdue_count = (datetime.now() - reminder.next_date).days
        cycles_to_skip = (days_overdue_count // reminder.frequency) + 1
        
        # Advance to next occurrence
        reminder.next_date = reminder.next_date + timedelta(days=reminder.frequency * cycles_to_skip)
        reminder.status = "scheduled"
        
        # Update note
        if reminder.note:
            reminder.note += f" | Auto-advanced {cycles_to_skip} cycle(s) on {datetime.now().strftime('%Y-%m-%d')}"
        else:
            reminder.note = f"Auto-advanced {cycles_to_skip} cycle(s) on {datetime.now().strftime('%Y-%m-%d')}"
        
        advanced += 1
    
    db.commit()
    return advanced
