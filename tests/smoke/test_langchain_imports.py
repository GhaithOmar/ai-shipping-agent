def test_imports():
    import langchain
    import langchain_community
    import langgraph

    from backend.settings import settings

    assert settings is not None
    assert langchain is not None
    assert langchain_community is not None
    assert langgraph is not None
