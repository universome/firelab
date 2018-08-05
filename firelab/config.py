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

        for key, value in config:
            self.set(key, value)

    def set(self, key, value):
        assert type(key) is str

        if type(value) is dict:
            setattr(self, key, Config(value))
        elif type(value) is list or type(value) is tuple:
            assert len(set([type(el) for el in value])) == 1, HOMOGENOUS_ARRAY_MSG

            if type(value[0]) is dict:
                # TODO: We should check types recursively
                setattr(self, key, tuple(Config(el) for el in value))
            else:
                setattr(self, key, tuple(value))
        elif type(value) is int or type(value) is float or type(value) is str:
            setattr(self, key, value)
        else:
            raise TypeError("Unsupported type for key {}: {}. "
                            "Value is {}".format(key, type(value), value))

    def __getattribute__(self, name):
        if not hasattr(self, name):
            return None
        else:
            return super(Config, self).__getattribute__(name)

    def __setattr__(self, name, value):
        assert not hasattr(self, name), IMMUTABILITY_ERROR_MSG

        super(Config, self).__setattr__(name, value)

    def __delattr__(self, name):
        assert not hasattr(self, name), IMMUTABILITY_ERROR_MSG
