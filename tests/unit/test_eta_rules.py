# tests/unit/test_eta_rules.py

import pytest
import os

try:
    from backend.tools.estimate_eta import estimate_eta  # type: ignore
except Exception:
    pytestmark = pytest.mark.skip(reason="estimate_eta module not present")

# Only after imports, set env defaults
os.environ.setdefault("AGENT_OFFLINE", "1")


def test_eta_gcc_intra_region_standard():
    out = estimate_eta("JO", "AE", "standard")
    # Expect essential keys and reasonable ranges
    assert out["origin_cc"] == "JO" and out["dest_cc"] == "AE"
    assert out["service_level"] == "standard"
    assert "eta_business_days" in out
    assert "min" in out["eta_business_days"] and "max" in out["eta_business_days"]
    assert 1 <= out["eta_business_days"]["min"] <= out["eta_business_days"]["max"] <= 15
