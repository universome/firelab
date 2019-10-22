import sys; sys.path.append('.')
from firelab.config import Config


def test_getter():
    assert Config({}).get("a") == None
    assert Config({"a": 2}).get("a") == 2
    assert Config({"a": 2}).get("b", 3) == 3
    assert Config({"a": {"b": 4}}).get("a.b") == 4
    assert Config({"a": {"b": 4}}).get("a.c", 5) == 5


def test_setter():
    assert Config({"a": 3}).a == 3
    assert Config({"a": 3, "b": {"c": 4}}).b.c == 4
    assert Config({"a.b": 3}).a.b == 3

    config = Config({"a.b": 3, "a.e": 4})
    config.set("a.c.d", 5)
    config.set("e.e", [1, 2, 3])

    assert config.a.b == 3
    assert config.a.e == 4
    assert config.a.c.d == 5
    assert config.e.e == tuple([1, 2, 3])



def test_overwrite():
    assert Config({}).overwrite(Config({"a": 3})).a == 3
    assert Config({"a": 2}).overwrite(Config({"a": 3})).a == 3
    assert Config({"b": 4}).overwrite(Config({"a": 3})).a == 3
    assert Config({"b": 4}).overwrite(Config({"a": 3})).b == 4
    assert Config({"a": {"b": 3}}).overwrite(Config({"b": 3})).b == 3
    assert Config({"a": {"b": 4}}).overwrite(Config({"b": 3})).a.b == 4
    assert Config({"a": {"b": 4}}).overwrite(Config({"a": {"c": 5}})).a.b == 4
    assert Config({"a": {"b": 4}}).overwrite(Config({"a": {"c": 5}})).a.c == 5
