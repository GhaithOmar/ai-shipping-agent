import re
from typing import List, Dict, Tuple, Optional

# Accept both underscored and compact aliases (case-insensitive)
CARRIER_ALIASES = [
    "Shipping_A", "Shipping_B", "Shipping_C",
    "ShippingA", "ShippingB", "ShippingC",
]

# Tokens that LOOK like tracking IDs, including internal hyphens/spaces.
# Example matches: "1Z-999-AA1-0123-4567-84", "123 456 789 012", "AB12-345678"
# Groups of A–Z0–9 blocks separated by hyphens/spaces, but the **first block must contain a digit**.
# This prevents matching words like "Track" before the real ID.
# Each block must include at least one digit; prevents trailing words like "with" being captured.
SEPARATED_ID_RE = re.compile(
    r"\b([A-Z0-9]*\d[A-Z0-9]*(?:[-\s][A-Z0-9]*\d[A-Z0-9]*){0,8})\b",
    re.IGNORECASE
)



# Obvious non-ID patterns to ignore
DATE_RE     = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")  # e.g., 2025-08-27
TIME_RE     = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")
INVOICE_RE  = re.compile(r"\b(?:INV|ORDER|PO|REF)[-:\s]?\d{3,}\b", re.IGNORECASE)
PUNCT_ONLY  = re.compile(r"^[\W_]+$")

def _normalize(token: str) -> str:
    s = token or ""
    # drop a leading word + separators if it appears before the first digit (e.g., "Track 1Z-...")
    s = re.sub(r'^[A-Za-z]+(?:[\s-]+)(?=\d)', '', s)
    return re.sub(r"[-\s]", "", s).upper()



def _looks_like_id(raw: str) -> Tuple[bool, Optional[str]]:
    """Heuristics to reduce false positives; returns (ok, normalized_id)."""
    if not raw or PUNCT_ONLY.match(raw):
        return False, None
    if DATE_RE.search(raw) or TIME_RE.search(raw) or INVOICE_RE.search(raw):
        return False, None

    norm = _normalize(raw)

    # Length bounds after normalization (tweak if you meet real formats)
    if not (8 <= len(norm) <= 22):
        return False, None

    # Must contain digits (avoid matching pure alphabetic words)
    if not any(ch.isdigit() for ch in norm):
        return False, None

    # Filter short pure-digit strings (often years/short refs)
    if norm.isdigit() and len(norm) < 8:
        return False, None

    return True, norm

def parse_tracking(text: str) -> Dict:
    """
    Extract carrier alias (if present) and normalize potential tracking IDs.
    Returns:
      {
        "carrier": "Shipping_A" | None,
        "ids":      [normalized_ids],
        "ids_raw":  [original_matched_tokens],
        "notes":    str|None
      }
    """
    carrier = None
    for alias in CARRIER_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", text or "", flags=re.IGNORECASE):
            carrier = alias if "Shipping_" in alias else alias.replace("Shipping", "Shipping_")
            break

    candidates = []
    for m in SEPARATED_ID_RE.finditer(text or ""):
        token = m.group(1)
        # Need at least one digit in the token to consider it a tracking-like string
        if re.search(r"\d", token):
            candidates.append(token)

    ids: List[str] = []
    ids_raw: List[str] = []

    for tok in candidates:
        ok, norm = _looks_like_id(tok)
        if not ok or norm is None:
            continue
        if norm not in ids:
            ids.append(norm)
            ids_raw.append(tok)

    notes = None
    if not carrier and not ids:
        notes = "no obvious carrier/ids found"

    return {"carrier": carrier, "ids": ids, "ids_raw": ids_raw, "notes": notes}
