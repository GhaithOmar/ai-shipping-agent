from backend.settings import settings

def test_settings_defaults():
    # sanity check a few defaults
    assert settings.qdrant_collection == "shipping_kb"
    assert isinstance(settings.agent_enable, bool)
    assert settings.agent_top_k > 0
