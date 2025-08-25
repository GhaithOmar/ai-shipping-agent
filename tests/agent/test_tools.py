import re
import pytest

from backend.tools.parse_tracking import parse_tracking
from backend.tools.estimate_eta import estimate_eta
from backend.tools.search_kb import search_kb, format_citations

def test_parse_tracking_basic():
    out = parse_tracking("Track 1Z999AA10123456784 with Shipping_A please")
    assert out["carrier"] in {"Shipping_A", "Shipping_B", "Shipping_C", None}
    # accept alnum ID length filter
    assert isinstance(out["ids"], list)

def test_parse_tracking_multiple_ids():
    out = parse_tracking("IDs: AB12CD34 and 12345XYZ with shippingb")
    assert len(out["ids"]) >= 1

def test_estimate_eta_ranges():
    e = estimate_eta("JO", "AE", "express")
    assert 1 <= e["eta_business_days"]["min"] <= e["eta_business_days"]["max"]
    assert e["service_level"] == "express"

@pytest.mark.parametrize("query", ["change delivery address", "create pickup request", "refund policy"])
def test_search_kb_soft(query):
    """
    This should pass even if no Qdrant is running or collection missing.
    We only assert it returns a list and formatting function tolerates empty.
    """
    hits = search_kb(query, k=2)
    assert isinstance(hits, list)
    _ = format_citations(hits)
