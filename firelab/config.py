"""
This is a config file. We keep it in global,
so we do not need to pass params across functions and models
"""
import os
import yaml
import argparse
from hashlib import sha256
from typing import List, Any, Dict


CONFIG_ARG_PREFIX = '--config.'
ALLOWED_LIST_SEPARATORS = [',', ' ', '|', '-']
ALLOWED_LIST_OPENERS = ['[', '(', '{']
ALLOWED_LIST_CLOSERS = [']', ')', '}']


class Config:
    @classmethod
    def load(cls, config_path: os.PathLike, frozen: bool=True) -> "Config":
        with open(config_path, "r", encoding="utf-8") as config_file:
            return Config.load_from_string(config_file, frozen=frozen)

    @classmethod
    def load_from_string(cls, config_string: str, frozen: bool=True) -> "Config":
        return Config(yaml.safe_load(config_string), frozen=frozen)

    @classmethod
    def read_from_cli(cls, should_infer_type: bool=True, config_arg_prefix: str=CONFIG_ARG_PREFIX) -> "Config":
        """Reads config args from the CLI and converts them to a config"""
        _, config_args = argparse.ArgumentParser().parse_known_args()

        # Filtering out those args that do not start with `config_arg_prefix`
        config = {c: config_args[i+1] for i, c in enumerate(config_args) if c.startswith(config_arg_prefix)}

        # Extracting true names (i.e. removing the prefix)
        config = {c[len(config_arg_prefix):]: v for c, v in config.items()}

        if should_infer_type:
            config = {c: infer_type_and_convert(v) for c, v in config.items()}

        return Config(config)

    def __init__(self, config, frozen: bool=True):
        assert type(config) is dict

        self._keys = set()
        self.is_frozen = frozen

        for key in config:
            self.set(key, config[key])

    def freeze(self):
        self.is_frozen = True

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

    def __getitem__(self, key: str) -> Any:
        assert self.has(key), f'Key {key} is missing in the config'

        return self.get(key)

    def set(self, attr_path, value):
        """Sets value to the config (if it was not set before)"""
        assert type(attr_path) is str

        curr_config = self
        attr_path = attr_path.split('.')
        attr_parent_path = '.'.join(attr_path[:-1]) # "a.b.c.d" => "a.b.c"
        attr_name = attr_path[-1]

        if len(attr_parent_path) > 0:
            if not self.has(attr_parent_path):
                self._create_path(attr_parent_path)

            curr_config = self.get(attr_parent_path)
        if type(value) is dict:
            setattr(curr_config, attr_name, Config(value, frozen=self.is_frozen))
        elif type(value) is Config:
            setattr(curr_config, attr_name, Config(value.to_dict(), frozen=self.is_frozen))
        elif type(value) is list or type(value) is tuple:
            # TODO: maybe we should put everything in list? tuples look weird
            if len(value) == 0:
                setattr(curr_config, attr_name, tuple())
            else:
                assert len(set([type(el) for el in value])) == 1, homogenous_array_message(value)

                if type(value[0]) is dict:
                    # TODO: We should check types recursively
                    setattr(curr_config, attr_name, tuple(Config(el, frozen=self.is_frozen) for el in value))
                else:
                    setattr(curr_config, attr_name, tuple(value))
        elif type(value) in [int, float, str, bool]:
            setattr(curr_config, attr_name, value)
        else:
            raise TypeError("Unsupported type for attr_name \"{}\": {}. "
                            "Value is {}".format(attr_name, type(value), value))

        curr_config._keys.add(attr_name)

    def _create_path(self, attr_path: str):
        """Creates attributes recursively, filled with empty configs"""
        attr_path = attr_path.split('.')
        curr_config = self

        for attr in attr_path:
            if not curr_config.has(attr): curr_config.set(attr, Config({}, frozen=self.is_frozen))

            curr_config = curr_config.get(attr)

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
        assert not hasattr(self, name) or not self.is_frozen, \
            f'You cannot change attributes (tried to change {name}), because config is frozen.'

        super(Config, self).__setattr__(name, value)

    def __str__(self):
        return yaml.safe_dump(self.to_dict(), default_flow_style=False)

    def __repr__(self):
        return str(self)

    def __delattr__(self, name):
        if self.is_frozen:
            # TODO: not sure if this is the right exception cls :|
            raise PermissionError("Config is frozen.")

        raise NotImplementedError('Attribute deletion is not implemented yet :|')

    def to_dict(self) -> Dict:
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

    def overwrite(self, config: "Config") -> "Config":
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

        return Config(result, frozen=self.is_frozen)

    def clone(self, frozen: bool=True) -> "Config":
        return Config(self.to_dict(), frozen=frozen)

    def compute_hash(self, size: int=10) -> str:
        return sha256(str(self).encode('utf-8')).hexdigest()[:size]


def homogenous_array_message(array:List) -> str:
    return f"You can provide only homogenous arrays. Array {array} has values of different type!"


def infer_type_and_convert(value:str) -> Any:
    """
    Chances are high that this function should never exist...
    It tries to get a proper type and converts the value to it.
    """
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.isdigit():
        return int(value)
    elif is_float(value):
        return float(value)
    elif is_list(value):
        if has_list_closers(value): value = value[1:-1]

        separator = next((s for s in ALLOWED_LIST_SEPARATORS if s in value), ',')
        value = [infer_type_and_convert(x) for x in value.split(separator) if len(x) > 0]

        return value
    else:
        return value


def is_list(value: str) -> bool:
    """A dirty function that checks if the value looks like list"""
    return is_separated(value) or has_list_closers(value)


def is_separated(value: str) -> bool:
    return any((s in value) for s in ALLOWED_LIST_SEPARATORS)


def has_list_closers(value: str) -> bool:
    try:
        return (ALLOWED_LIST_OPENERS.index(value[0]) == ALLOWED_LIST_CLOSERS.index(value[-1]))
    except:
        return False


def is_float(value: Any) -> bool:
    """One more dirty function: it checks if the string is float."""
    try:
        float(value)
        return True
    except ValueError:
        return False
