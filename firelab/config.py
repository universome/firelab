"""
This is a config file. We keep it in global,
so we do not need to pass params across functions and models
"""
# TODO: is it really a good thing to keep it in global
# instead of passing across functions?

IMMUTABILITY_ERROR_MSG = "Config properties are immutable"
HOMOGENOUS_ARRAY_MSG = "Config supports only homogenous arrays"

class Config:
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
        elif type(value) is list or type(value) is tuple:
            # TODO: maybe we should put everything in list? tuples look wierd

            if len(value) == 0:
                setattr(self, key, tuple())
            else:
                assert len(set([type(el) for el in value])) == 1, HOMOGENOUS_ARRAY_MSG

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

    def has(self, key):
        """
        Checks, if the given key was set
        (note, that it can be set to None)
        """
        return hasattr(self, key)

    def __setattr__(self, name, value):
        assert not hasattr(self, name), IMMUTABILITY_ERROR_MSG

        super(Config, self).__setattr__(name, value)

    def __delattr__(self, name):
        # TODO: not sure if this is the right exception cls :|
        raise PermissionError(IMMUTABILITY_ERROR_MSG)

    def to_dict(self):
        result = {}

        for key in self.keys():
            if isinstance(self.get(key), Config):
                result[key] = self.get(key).to_dict()
            else:
                result[key] = self.get(key)

        return result
