# tests/unit/test_eta_rules.py
import os
import importlib
import pytest

os.environ.setdefault("AGENT_OFFLINE", "1")

eta_mod = importlib.util.find_spec("backend.tools.estimate_eta")
if eta_mod is None:
    pytest.skip("estimate_eta module not present", allow_module_level=True)

from backend.tools.estimate_eta import estimate_eta  # type: ignore


def test_eta_gcc_intra_region_standard():
    out = estimate_eta("JO", "AE", "standard")
    # Expect essential keys and reasonable ranges
    assert out["origin_cc"] == "JO" and out["dest_cc"] == "AE"
    assert out["service_level"] == "standard"
    assert "eta_business_days" in out and "min" in out["eta_business_days"] and "max" in out["eta_business_days"]
    assert 1 <= out["eta_business_days"]["min"] <= out["eta_business_days"]["max"] <= 15
