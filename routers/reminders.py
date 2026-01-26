# backend/routers/reminders.py
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta

from dependencies import get_db
from schemas import ReminderCreate, ReminderUpdate, ReminderOut
from models import Reminder, Customer, Sale

router = APIRouter(prefix="/reminders", tags=["Reminders"])


@router.post("", response_model=ReminderOut)
def create_reminder(payload: ReminderCreate, db: Session = Depends(get_db)):
    if not payload.customer_id and not payload.custom_name:
        raise HTTPException(400, "Either customer_id or custom_name is required")

    # ⭐ NEW: If creating manual reminder for a customer, delete auto-generated ones
    if payload.customer_id:
        # Find and delete auto-generated reminders (those with "Auto-" in note)
        auto_reminders = (
            db.query(Reminder)
            .filter(
                Reminder.customer_id == payload.customer_id,
                Reminder.status.in_(["pending", "scheduled"]),
                Reminder.note.like("%Auto-%")  # Auto-generated or Auto-created
            )
            .all()
        )
        
        for auto_reminder in auto_reminders:
            db.delete(auto_reminder)
        
        if auto_reminders:
            db.commit()  # Commit deletion before creating new one

    r = Reminder(
        customer_id=payload.customer_id,
        custom_name=payload.custom_name,
        reason=payload.reason,
        frequency=payload.frequency,
        next_date=payload.next_date,
        note=payload.note,
        status=payload.status,
    )

    db.add(r)
    db.commit()
    db.refresh(r)

    return r


@router.get("")
def list_reminders(db: Session = Depends(get_db)):
    reminders = (
        db.query(Reminder)
        .order_by(Reminder.next_date.asc())
        .all()
    )

    profiled = []
    customs = []

    for r in reminders:
        name = None
        last_sale_date = None
        activity_status = None
        
        if r.customer_id:
            cust = db.query(Customer).filter(Customer.id == r.customer_id).first()
            if cust:
                name = cust.name
                activity_status = cust.activity_status
                
                # Get last sale date
                last_sale = (
                    db.query(Sale)
                    .filter(Sale.customer_id == r.customer_id)
                    .order_by(Sale.date.desc())
                    .first()
                )
                if last_sale:
                    last_sale_date = last_sale.date.isoformat() if last_sale.date else None

        item = {
            "id": r.id,
            "customer_id": r.customer_id,
            "customer_name": name,
            "custom_name": r.custom_name,
            "reason": r.reason,
            "frequency": r.frequency,
            "next_date": r.next_date,
            "note": r.note,
            "status": r.status,
            "created_at": r.created_at,
            "last_sale_date": last_sale_date,
            "activity_status": activity_status,
        }

        if r.customer_id:
            profiled.append(item)
        else:
            customs.append(item)

    return {"profiled": profiled, "customs": customs}


@router.get("/due/today")
def get_today_reminders(db: Session = Depends(get_db)):
    """Get reminders due today (for bell icon) - matches reminders page 'Today' filter"""
    try:
        # Get ALL pending/scheduled reminders and filter in Python (same as frontend does)
        # This ensures consistency with the reminders page "Today" filter
        reminders = (
            db.query(Reminder)
            .filter(Reminder.status.in_(["pending", "scheduled"]))
            .order_by(Reminder.next_date.asc())
            .all()
        )
        
        if not reminders:
            return []
        
        # Filter for today (same logic as frontend ReminderTable)
        from datetime import date
        today = date.today()
        
        result = []
        for r in reminders:
            try:
                # Check if reminder is for today
                if r.next_date:
                    reminder_date = r.next_date.date() if hasattr(r.next_date, 'date') else r.next_date
                    if reminder_date != today:
                        continue
                else:
                    continue
                
                customer_name = None
                last_sale_date = None
                activity_status = None
                
                if r.customer_id:
                    customer = db.query(Customer).filter(Customer.id == r.customer_id).first()
                    if customer:
                        customer_name = customer.name
                        activity_status = customer.activity_status
                        
                        # Get last sale date
                        last_sale = (
                            db.query(Sale)
                            .filter(Sale.customer_id == r.customer_id)
                            .order_by(Sale.date.desc())
                            .first()
                        )
                        if last_sale:
                            last_sale_date = last_sale.date.isoformat() if last_sale.date else None
                
                result.append({
                    "id": r.id,
                    "customer_id": r.customer_id,
                    "customer_name": customer_name,
                    "custom_name": r.custom_name,
                    "reason": r.reason or "delivery",
                    "frequency": r.frequency if r.frequency is not None else 0,
                    "next_date": r.next_date.isoformat() if r.next_date else datetime.now().isoformat(),
                    "note": r.note or "",
                    "status": r.status or "pending",
                    "created_at": r.created_at.isoformat() if r.created_at else datetime.now().isoformat(),
                    "last_sale_date": last_sale_date,
                    "activity_status": activity_status,
                })
            except Exception as e:
                print(f"Error serializing reminder {r.id}: {str(e)}")
                continue
        
        return result
    except Exception as e:
        print(f"Error in get_today_reminders: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


@router.get("/overdue")
async def get_overdue_reminders(db: Session = Depends(get_db)):
    """Get all overdue reminders (past due date and still pending/scheduled)"""
    from datetime import timezone
    
    try:
        now = datetime.now(timezone.utc)
        
        reminders = (
            db.query(Reminder)
            .filter(Reminder.next_date < now)
            .filter(Reminder.status.in_(["pending", "scheduled"]))
            .order_by(Reminder.next_date.asc())
            .all()
        )
        
        if not reminders:
            return []
        
        result = []
        for r in reminders:
            try:
                customer_name = None
                last_sale_date = None
                activity_status = None
                
                if r.customer_id:
                    customer = db.query(Customer).filter(Customer.id == r.customer_id).first()
                    if customer:
                        customer_name = customer.name
                        activity_status = customer.activity_status
                        
                        # Get last sale date
                        last_sale = (
                            db.query(Sale)
                            .filter(Sale.customer_id == r.customer_id)
                            .order_by(Sale.date.desc())
                            .first()
                        )
                        if last_sale:
                            last_sale_date = last_sale.date.isoformat() if last_sale.date else None
                
                result.append({
                    "id": r.id,
                    "customer_id": r.customer_id,
                    "customer_name": customer_name,
                    "custom_name": r.custom_name,
                    "reason": r.reason or "delivery",
                    "frequency": r.frequency if r.frequency is not None else 0,
                    "next_date": r.next_date.isoformat() if r.next_date else now.isoformat(),
                    "note": r.note or "",
                    "status": r.status or "pending",
                    "created_at": r.created_at.isoformat() if r.created_at else now.isoformat(),
                    "last_sale_date": last_sale_date,
                    "activity_status": activity_status,
                })
            except Exception:
                continue
        
        return result
        
    except Exception:
        return []


@router.get("/{reminder_id}", response_model=ReminderOut)
def get_reminder(reminder_id: int, db: Session = Depends(get_db)):
    r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not r:
        raise HTTPException(404, "Reminder not found")
    return r


@router.put("/{reminder_id}", response_model=ReminderOut)
def update_reminder(reminder_id: int, payload: ReminderUpdate, db: Session = Depends(get_db)):
    r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not r:
        raise HTTPException(404, "Reminder not found")

    for key, value in payload.dict(exclude_unset=True).items():
        setattr(r, key, value)

    db.commit()
    db.refresh(r)
    return r


@router.delete("/{reminder_id}")
def delete_reminder(reminder_id: int, db: Session = Depends(get_db)):
    r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not r:
        raise HTTPException(404, "Reminder not found")

    db.delete(r)
    db.commit()

    return {"message": "Reminder deleted successfully"}


@router.post("/{reminder_id}/status")
def update_reminder_status(
    reminder_id: int,
    payload: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Update reminder status (skip, cancel, etc.)
    
    When status is "skipped":
    - Marks current reminder as skipped
    - Creates a new reminder for next occurrence (if frequency > 0)
    """
    try:
        r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
        if not r:
            raise HTTPException(404, "Reminder not found")

        status = payload.get("status")
        if not status:
            raise HTTPException(400, "Status is required in request body")

        # ⭐ UPDATED: Skip current and create next occurrence
        if status == "skipped" and r.frequency and int(r.frequency) > 0:
            # Mark current as skipped
            r.status = "skipped"
            
            # Replace note instead of appending
            r.note = f"Skipped on {datetime.now().strftime('%Y-%m-%d')}"
            
            db.add(r)
            db.commit()
            
            # Create new reminder for next occurrence
            frequency_days = int(r.frequency)
            next_date = r.next_date + timedelta(days=frequency_days)
            
            new_reminder = Reminder(
                customer_id=r.customer_id,
                custom_name=r.custom_name,
                reason=r.reason,
                frequency=r.frequency,
                next_date=next_date,
                note=f"Created after skip",
                status="scheduled",
            )
            db.add(new_reminder)
            db.commit()
            db.refresh(new_reminder)
            
            return {
                "id": r.id,
                "status": r.status,
                "next_date": r.next_date.isoformat() if r.next_date else None,
                "next_reminder_id": new_reminder.id,
                "next_reminder_date": new_reminder.next_date.isoformat(),
                "message": f"Reminder skipped and next occurrence created for {new_reminder.next_date.strftime('%Y-%m-%d')}"
            }
        else:
            # For other statuses or one-time reminders, just update status
            r.status = status
            
            if status == "skipped":
                r.note = f"Skipped on {datetime.now().strftime('%Y-%m-%d')}"
            
            db.commit()
            db.refresh(r)
            
            return {
                "id": r.id,
                "status": r.status,
                "next_date": r.next_date.isoformat() if r.next_date else None,
                "message": f"Reminder status updated to {r.status}"
            }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating reminder status: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Failed to update status: {str(e)}")


@router.post("/{reminder_id}/done")
def mark_reminder_done(reminder_id: int, db: Session = Depends(get_db)):
    """
    ⭐ Mark reminder as DONE (skipped/cancelled)
    
    This is for when you want to skip this reminder without making a sale.
    The reminder will be marked as "skipped" and won't show up anymore.
    """
    try:
        r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
        if not r:
            raise HTTPException(404, "Reminder not found")

        # Mark as skipped
        r.status = "skipped"
        
        # Replace note
        r.note = "Marked as done"
        
        db.commit()
        db.refresh(r)
        
        return {
            "message": "Reminder marked as done (skipped)",
            "reminder_id": r.id,
            "status": r.status
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error marking reminder as done: {str(e)}")
        raise HTTPException(500, f"Failed to mark as done: {str(e)}")


@router.post("/{reminder_id}/complete")
def complete_reminder(reminder_id: int, db: Session = Depends(get_db)):
    """
    ⭐ Mark reminder as COMPLETED and create next reminder
    
    This should be called automatically when a sale is made.
    It marks the current reminder as completed and creates the next one based on frequency.
    
    Note: This is called automatically by the sale creation endpoint.
    """
    try:
        r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
        if not r:
            raise HTTPException(404, "Reminder not found")

        # Mark current as completed
        r.status = "completed"
        db.commit()
        db.refresh(r)

        # If recurring (frequency > 0), create next reminder
        # Convert frequency to int to avoid type comparison errors
        freq = int(r.frequency) if r.frequency else 0
        if freq > 0:
            next_date = datetime.now() + timedelta(days=freq)
            
            next_reminder = Reminder(
                customer_id=r.customer_id,
                custom_name=r.custom_name,
                reason=r.reason,
                frequency=r.frequency,
                next_date=next_date,
                note=f"Auto-created after sale on {datetime.now().strftime('%Y-%m-%d')}",
                status="scheduled",
            )
            db.add(next_reminder)
            db.commit()
            db.refresh(next_reminder)
            
            return {
                "message": "Reminder completed and next reminder created",
                "completed_id": r.id,
                "next_reminder_id": next_reminder.id,
                "next_date": next_reminder.next_date.isoformat()
            }

        return {
            "message": "Reminder completed (one-time)",
            "completed_id": r.id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error completing reminder: {str(e)}")
        raise HTTPException(500, f"Failed to complete reminder: {str(e)}")


@router.post("/{reminder_id}/move-tomorrow")
def move_reminder_to_tomorrow(reminder_id: int, db: Session = Depends(get_db)):
    """Move reminder to tomorrow (same time)"""
    try:
        r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
        if not r:
            raise HTTPException(404, "Reminder not found")

        # Get tomorrow's date with same time
        current_date = r.next_date
        tomorrow = current_date + timedelta(days=1)
        
        # Update reminder
        old_date = r.next_date.strftime('%Y-%m-%d')
        r.next_date = tomorrow
        r.status = "pending"
        
        # Update note
        if r.note:
            r.note += f" | Moved to tomorrow on {datetime.now().strftime('%Y-%m-%d')}"
        else:
            r.note = f"Moved to tomorrow on {datetime.now().strftime('%Y-%m-%d')}"
        
        db.commit()
        db.refresh(r)
        
        return {
            "message": f"Reminder moved from {old_date} to tomorrow",
            "reminder_id": r.id,
            "new_date": r.next_date.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error moving reminder: {str(e)}")
        raise HTTPException(500, f"Failed to move reminder: {str(e)}")


@router.post("/{reminder_id}/advance")
def advance_next_date(reminder_id: int, db: Session = Depends(get_db)):
    r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not r:
        raise HTTPException(404, "Reminder not found")

    freq = int(r.frequency) if r.frequency else 0
    if freq == 0:
        raise HTTPException(400, "Cannot advance a one-time reminder (frequency = 0).")

    r.next_date = r.next_date + timedelta(days=freq)
    r.status = "scheduled"
    db.commit()
    db.refresh(r)
    return r


@router.delete("/cleanup")
def cleanup_old_reminders(db: Session = Depends(get_db)):
    """Only delete completed/cancelled reminders older than 30 days"""
    month_ago = datetime.now() - timedelta(days=30)
    removed = (
        db.query(Reminder)
        .filter(Reminder.next_date < month_ago)
        .filter(Reminder.status.in_(["completed", "cancelled"]))
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"deleted": removed, "note": "Only completed/cancelled reminders older than 30 days were deleted"}


@router.post("/generate-smart")
def generate_smart_reminders_endpoint(
    max_inactive_days: int = 60,
    db: Session = Depends(get_db)
):
    """
    Analyze customer patterns and auto-generate reminders for DELIVERY customers who are due
    
    Only creates reminders for customers with:
    - activity_status = 'active'
    - delivery_type = 'delivery' (not self-pickup)
    
    Parameters:
    - max_inactive_days: Maximum days since last sale to consider customer active (default: 60)
    
    Returns:
    - created: Number of reminders created
    - skipped: Number of customers skipped (already have reminders or not due yet)
    - inactive: Number of customers marked as inactive (no purchase in max_inactive_days)
    - no_pattern: Number of customers without enough data
    """
    from services.smart_reminder_service import generate_smart_reminders
    result = generate_smart_reminders(db, max_inactive_days)
    return result


@router.post("/advance-overdue")
def advance_overdue_reminders_manual(db: Session = Depends(get_db)):
    """
    ⭐ MANUAL: Advance all overdue reminders to TODAY
    
    This moves ALL yesterday's and older overdue reminders to today (same time).
    Use this button when you want to manually advance overdue reminders.
    
    Behavior:
    - All overdue reminders (yesterday and older) → Move to TODAY
    - Status changed to "pending"
    - Only affects pending/scheduled reminders
    
    Note: This does NOT skip cycles, just moves them to today.
    """
    now = datetime.now()
    today = now.date()
    
    print(f"🔍 ADVANCE OVERDUE: Today is {today}")
    
    # Find all overdue reminders (before today)
    overdue_reminders = (
        db.query(Reminder)
        .filter(
            func.date(Reminder.next_date) < today,
            Reminder.status.in_(["pending", "scheduled"])
        )
        .all()
    )
    
    print(f"   Found {len(overdue_reminders)} overdue reminders")
    
    if not overdue_reminders:
        return {
            "message": "No overdue reminders found",
            "advanced": 0
        }
    
    advanced_count = 0
    
    for reminder in overdue_reminders:
        old_date = reminder.next_date
        
        # Move to today (same time as original)
        original_time = reminder.next_date.time()
        reminder.next_date = datetime.combine(today, original_time)
        
        # Preserve timezone if original had one
        if old_date.tzinfo is not None:
            reminder.next_date = reminder.next_date.replace(tzinfo=old_date.tzinfo)
        
        reminder.status = "pending"
        
        print(f"   Reminder {reminder.id}: {old_date} → {reminder.next_date}")
        
        # Replace note instead of appending
        reminder.note = f"Advanced to today"
        
        db.add(reminder)
        advanced_count += 1
    
    db.commit()
    
    print(f"   ✅ Advanced {advanced_count} reminders to today")
    
    return {
        "message": f"Advanced {advanced_count} overdue reminder(s) to today",
        "advanced": advanced_count
    }
    return {
        "message": f"Advanced {advanced_count} overdue reminder(s) to today",
        "advanced": advanced_count
    }


@router.get("/customer-pattern/{customer_id}")
def get_customer_pattern(customer_id: int, db: Session = Depends(get_db)):
    """Get customer's purchase pattern analysis"""
    from services.smart_reminder_service import analyze_customer_pattern
    
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(404, "Customer not found")
    
    avg_days = analyze_customer_pattern(customer_id, db)
    
    # Get last sale
    last_sale = (
        db.query(Sale)
        .filter(Sale.customer_id == customer_id)
        .order_by(Sale.date.desc())
        .first()
    )
    
    return {
        "customer_id": customer_id,
        "customer_name": customer.name,
        "average_days_between_orders": avg_days,
        "last_sale_date": last_sale.date if last_sale else None,
        "recommendation": f"Create reminder every {avg_days} days" if avg_days else "Not enough data (need at least 2 sales)"
    }
