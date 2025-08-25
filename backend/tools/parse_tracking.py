import re
from typing import List, Dict

CARRIER_ALIASES = ["Shipping_A", "Shipping_B", "Shipping_C", "ShippingA", "ShippingB", "ShippingC"]

# Tracking IDs: allow digits/uppercase letters, length 6–20 (adjust if you have stricter formats)
TRACKING_RE = re.compile(r"\b([A-Z0-9]{6,20})\b")

def parse_tracking(text: str) -> Dict:
    """
    Extracts carrier alias (if present) and tracking IDs from free text.
    Returns {carrier: str|None, ids: [str], notes: str|None}
    """
    carrier = None
    for alias in CARRIER_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE):
            carrier = alias.replace("Shipping", "Shipping_") if "Shipping_" not in alias else alias
            break

    # exclude common non-IDs like years, short numbers
    candidates = [m.group(1) for m in TRACKING_RE.finditer(text)]
    ids: List[str] = []
    for c in candidates:
        if c.isdigit() and len(c) <= 6:   # avoid short years/invoice nos
            continue
        ids.append(c)

    ids = list(dict.fromkeys(ids))  # de-dup preserve order
    return {"carrier": carrier, "ids": ids, "notes": None if (carrier or ids) else "no obvious carrier/ids found"}
