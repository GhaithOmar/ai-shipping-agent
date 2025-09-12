import os
os.environ["AGENT_OFFLINE"] = "1"  # ensure offline in unit tests

try:
    # prefer the backend.tools module path used in your app
    from backend.tools.parse_tracking import parse_tracking
except Exception:
    # fallback if your layout differs locally
    from parse_tracking import parse_tracking  # type: ignore

def test_carrier_and_id_extraction():
    text = "Please track 12345678 with Shipping_A today."
    res = parse_tracking(text)
    assert res["carrier"] == "Shipping_A"
    assert "12345678" in res["ids"]

def test_alias_normalization_and_dedup():
    text = "ShippingA says AB12CD34 will arrive soon. Also AB12CD34 duplicated."
    res = parse_tracking(text)
    assert res["carrier"] == "Shipping_A"      # alias normalized
    assert res["ids"] == ["AB12CD34"]          # de-duplicated

def test_filters_short_numbers():
    text = "year 2024 and invoice 12345 are not tracking; 1234567 is."
    res = parse_tracking(text)
    assert "2024" not in res["ids"]
    assert "12345" not in res["ids"]
    assert "1234567" in res["ids"]

def test_no_hits_reports_note():
    text = "hello world"
    res = parse_tracking(text)
    assert res["carrier"] is None
    assert res["ids"] == []
    assert res["notes"]  # non-empty note explaining no obvious ids
