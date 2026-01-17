# backend/utils/helpers.py


def normalize_name(name: str) -> str:
    """Normalize customer name – trim + lowercase compare + title case storage."""
    clean = (name or "").strip()
    if not clean:
        return ""
    return clean.title()  # Example: 'adIl' → 'Adil'
