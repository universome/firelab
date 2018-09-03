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

    def get(self, name):
        """
            Safe getter (i.e. returns None instead of
            raising  exception in case of attribute is not set
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return None

    def set(self, key, value):
        """Setter with some validation"""

        assert type(key) is str

        if type(value) is dict:
            setattr(self, key, Config(value))
        elif type(value) is list or type(value) is tuple:
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
            raise TypeError("Unsupported type for key {}: {}. "
                            "Value is {}".format(key, type(value), value))

        self._keys.add(key)

    def keys(self):
        return self._keys

    def __setattr__(self, name, value):
        assert not hasattr(self, name), IMMUTABILITY_ERROR_MSG

        super(Config, self).__setattr__(name, value)

    def __delattr__(self, name):
        # TODO: not sure if this is the right exception cls :|
        raise PermissionError(IMMUTABILITY_ERROR_MSG)
