# tests/unit/test_rate_quote.py
import os

# keep everything offline/deterministic
os.environ.setdefault("AGENT_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    # prefer the canonical path in your repo
    from backend.tools.rate_quote import rate_quote
except Exception:
    # fallback if local layout differs
    from rate_quote import rate_quote  # type: ignore


def test_intra_country_basic():
    out = rate_quote(origin_cc="JO", dest_cc="JO", weight_kg=0.4)
    assert out["zone"] == "INTRA_COUNTRY"
    # first 0.5 kg base for INTRA_COUNTRY = 3.5 (see table), standard multiplier = 1.0
    assert out["price_usd_est"] == 3.5
    assert out["billable_weight_kg"] == 0.4


def test_intra_region_volumetric_vs_actual():
    # JO→AE are both in GCC → INTRA_REGION
    # actual = 0.6 kg, volumetric = (40*30*20)/5000 = 4.8 kg → billable = 4.8
    out = rate_quote(origin_cc="JO", dest_cc="AE", weight_kg=0.6, dims_cm=(40, 30, 20))
    assert out["zone"] == "INTRA_REGION"
    assert out["billable_weight_kg"] == 4.8
    # INTRA_REGION table: base 6.0 for first 0.5 kg + 2.0 per extra 0.5 kg
    # halves = ceil(4.8/0.5) = 10 → price = 6.0 + 2.0*(10-1) = 24.0
    assert out["price_usd_est"] == 24.0


def test_intercontinent_express_multiplier():
    # JO (GCC) → US (OTHER) → INTERCONTINENT, 1.2 kg actual, no dims
    out = rate_quote(origin_cc="JO", dest_cc="US", weight_kg=1.2, dims_cm=None, service_level="express")
    assert out["zone"] == "INTERCONTINENT"
    # halves = ceil(1.2/0.5) = 3
    # base 12.0 + 4.0*(3-1) = 20.0 then * express 1.6 = 32.0
    assert out["price_usd_est"] == 32.0
    assert out["service_level"] == "express"
