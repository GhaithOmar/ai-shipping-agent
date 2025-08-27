# backend/tools/rate_quote.py
from __future__ import annotations
from typing import Dict, Optional, Tuple

# Very simple, explainable rate table (USD).
# Base price per 0.5 kg for zones; we’ll infer zones from country codes roughly.
ZONE_BASE = {
    "INTRA_COUNTRY":  (3.5,  1.0),   # (base for first 0.5kg, per extra 0.5kg)
    "INTRA_REGION":   (6.0,  2.0),
    "INTERCONTINENT": (12.0, 4.0),
}

# Volumetric divisor (cm). Industry common: 5000–6000 for cm units.
VOL_DIV = 5000.0

GCC = {"AE","SA","KW","QA","BH","OM","JO"}
EU  = {"DE","FR","IT","ES","NL","SE","PL","BE","AT","DK","IE","PT","CZ","FI","RO","HU","GR"}

def _zone(o: str, d: str) -> str:
    o = (o or "").upper(); d = (d or "").upper()
    if o == d:
        return "INTRA_COUNTRY"
    def group(cc: str):
        if cc in GCC: return "GCC"
        if cc in EU:  return "EU"
        return "OTHER"
    go, gd = group(o), group(d)
    if go == gd and go in {"GCC","EU"}:
        return "INTRA_REGION"
    return "INTERCONTINENT"

def _billable_weight(kg: float, dims_cm: Optional[Tuple[float,float,float]] = None) -> float:
    actual = max(0.01, float(kg or 0))
    if dims_cm and all(x and x > 0 for x in dims_cm):
        L,W,H = dims_cm
        vol = (L * W * H) / VOL_DIV  # kg
        return max(actual, vol)
    return actual

def rate_quote(
    origin_cc: str,
    dest_cc: str,
    weight_kg: float,
    dims_cm: Optional[Tuple[float,float,float]] = None,
    service_level: Optional[str] = "standard"
) -> Dict:
    """
    Returns a rough, handbook-style quote with transparent math.
    """
    billable = _billable_weight(weight_kg, dims_cm)
    z = _zone(origin_cc, dest_cc)
    base_first, per_half = ZONE_BASE[z]

    # simple service multiplier
    mult = {"express": 1.6, "standard": 1.0, "economy": 0.8}.get((service_level or "standard").lower(), 1.0)

    # Price ladder: first 0.5 kg at base, then each 0.5 kg adds per_half
    import math
    halves = max(1, math.ceil(billable / 0.5))
    price = (base_first + per_half * (halves - 1)) * mult

    return {
        "origin_cc": (origin_cc or "").upper(),
        "dest_cc": (dest_cc or "").upper(),
        "service_level": (service_level or "standard").lower(),
        "billable_weight_kg": round(billable, 2),
        "zone": z,
        "price_usd_est": round(price, 2),
        "disclaimer": "Non-binding handbook estimate. Actual rates depend on carrier contracts, surcharges, and pickup address.",
        "assumptions": {
            "volumetric_divisor_cm": VOL_DIV,
            "pricing_unit": "per 0.5 kg",
            "tables": ZONE_BASE,
            "service_multiplier": {"express": 1.6, "standard": 1.0, "economy": 0.8},
        }
    }
