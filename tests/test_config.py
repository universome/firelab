from firelab.config import Config

def test_getter():
    assert Config({}).get("a") == None
    assert Config({"a": 2}).get("a") == 2
    assert Config({"a": 2}).get("b", 3) == 3
    assert Config({"a": {"b": 4}}).get("a.b") == 4
    assert Config({"a": {"b": 4}}).get("a.c", 5) == 5
