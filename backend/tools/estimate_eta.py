from typing import Dict, Optional

# naive business-day tables (tune later or source from docs)
SERVICE_BASE = {
    "standard": (3, 7),
    "express": (1, 3),
    "economy": (5, 12),
}

# region bumpers (very rough)
INTRA_COUNTRY = (-1, 0)      # slightly faster if same country
INTRA_REGION = (0, 1)        # e.g., EU↔EU, GCC↔GCC
INTERCONTINENTAL = (2, 5)    # slower

GCC = {"AE","SA","KW","QA","BH","OM","JO"}  # put JO here for regional tweak
EU  = {"DE","FR","IT","ES","NL","SE","PL","BE","AT","DK","IE","PT","CZ","FI","RO","HU","GR"}

def _region_group(cc: str) -> str:
    cc = (cc or "").upper()
    if cc in GCC: return "GCC"
    if cc in EU:  return "EU"
    return "OTHER"

def _region_bump(o: str, d: str):
    if o == d:
        return INTRA_COUNTRY
    ro, rd = _region_group(o), _region_group(d)
    if ro == rd and ro in {"GCC","EU"}:
        return INTRA_REGION
    return INTERCONTINENTAL

def estimate_eta(origin_cc: str, dest_cc: str, service_level: Optional[str] = "standard") -> Dict:
    """
    Super-simple ETA estimator in business days.
    service_level ∈ {standard, express, economy}
    """
    base = SERVICE_BASE.get((service_level or "standard").lower(), SERVICE_BASE["standard"])
    bump = _region_bump((origin_cc or "").upper(), (dest_cc or "").upper())
    lo = max(1, base[0] + bump[0])
    hi = max(lo, base[1] + bump[1])
    return {
        "origin_cc": (origin_cc or "").upper(),
        "dest_cc": (dest_cc or "").upper(),
        "service_level": (service_level or "standard").lower(),
        "eta_business_days": {"min": lo, "max": hi},
        "disclaimer": "Rough handbook estimate. Not live tracking.",
    }
