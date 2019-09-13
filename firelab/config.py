"""
This is a config file. We keep it in global,
so we do not need to pass params across functions and models
"""
import os
import yaml
from typing import List


class Config:
    @classmethod
    def load(cls, config_path:os.PathLike) -> "Config":
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = Config(yaml.safe_load(config_file))

        return config

    def __init__(self, config):
        assert type(config) is dict

        self._keys = set()

        for key in config:
            self.set(key, config[key])

    def get(self, attr_path:str, default_value=None):
        """
            Safe getter (i.e. returns None instead of
            raising  exception in case of attribute is not set

            Usage:
                - Config({}).get("a") # => None
                - Config({"a": 2}).get("a") # => 2
                - Config({"a": 2}).get("a", 3) # => 3
                - Config({"a": {"b": 4}}).get("a.b") # => 4
                - Config({"a": {"b": 4}}).get("a.c", 5) # => 5
        """
        curr_config = self
        attrs = attr_path.split('.')

        for attr_name in attrs:
            if hasattr(curr_config, attr_name):
                value = getattr(curr_config, attr_name)

                if attr_name == attrs[-1]:
                    return value
                elif isinstance(value, Config):
                    curr_config = value
                else:
                    break
            else:
                break

        return default_value

    def set(self, key, value):
        """Setter with some validation"""

        assert type(key) is str

        if type(value) is dict:
            setattr(self, key, Config(value))
        elif type(value) is Config:
            setattr(self, key, Config(value.to_dict()))
        elif type(value) is list or type(value) is tuple:
            # TODO: maybe we should put everything in list? tuples look wierd

            if len(value) == 0:
                setattr(self, key, tuple())
            else:
                assert len(set([type(el) for el in value])) == 1, homogenous_array_message(value)

                if type(value[0]) is dict:
                    # TODO: We should check types recursively
                    setattr(self, key, tuple(Config(el) for el in value))
                else:
                    setattr(self, key, tuple(value))
        elif type(value) in [int, float, str, bool]:
            setattr(self, key, value)
        else:
            raise TypeError("Unsupported type for key \"{}\": {}. "
                            "Value is {}".format(key, type(value), value))

        self._keys.add(key)

    def keys(self):
        return self._keys

    def has(self, attr_path:str):
        """
        Checks, if the given key was set
        (note, that it can be set to None)

        TODO: this code is almost identical to .get() method. Refactor.
        """
        curr_config = self
        attrs = attr_path.split('.')

        for attr_lvl_name in attrs:
            if hasattr(curr_config, attr_lvl_name):
                value = getattr(curr_config, attr_lvl_name)

                if attr_lvl_name == attrs[-1]:
                    return True
                elif isinstance(value, Config):
                    curr_config = value
                else:
                    break
            else:
                break

        return False

    def __setattr__(self, name, value):
        assert not hasattr(self, name), \
            f'You cannot change attributes (tried to change {name}), because config is immutable.'

        super(Config, self).__setattr__(name, value)

    def __str__(self):
        return yaml.safe_dump(self.to_dict(), default_flow_style=False)

    def __repr__(self):
        return str(self)

    def __delattr__(self, name):
        # TODO: not sure if this is the right exception cls :|
        raise PermissionError("Config is immutable.")

    def to_dict(self):
        result = {}

        for key in self.keys():
            if isinstance(self.get(key), Config):
                result[key] = self.get(key).to_dict()
            else:
                result[key] = self.get(key)

        return result

    def save(self, save_path:os.PathLike, parents:bool=True):
        """Saves config in the specified path"""
        if parents and not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

    def overwrite(self, config:"Config") -> "Config":
        """
        Overwrites current config with the provided one
        """
        result = self.to_dict()

        for key in config.keys():
            if key in result:
                if type(config.get(key)) is Config:
                    result[key] = Config(result[key]).overwrite(config.get(key))
                else:
                    result[key] = config.get(key)
            else:
                result[key] = config.get(key)

        return Config(result)



def homogenous_array_message(array:List) -> str:
    return f"You can provide only homogenous arrays. Array {array} has values of different type!"
