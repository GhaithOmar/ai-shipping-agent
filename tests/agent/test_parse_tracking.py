from backend.tools.parse_tracking import parse_tracking

def test_parse_hyphenated_and_spaced():
    out1 = parse_tracking("Track 1Z-999-AA1-0123-4567-84 with Shipping_A")
    assert "1Z999AA10123456784" in out1["ids"]
    assert out1["carrier"] == "Shipping_A"

    out2 = parse_tracking("status for 123 456 789 012")
    assert "123456789012" in out2["ids"]

    out3 = parse_tracking("My invoice INV-998877 and date 2025-08-27")
    assert out3["ids"] == []
