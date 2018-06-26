"""
This is a config file. We keep it in global,
so we do not need to pass params across functions and models
"""
# TODO: is it really a good thing to keep it in global
# instead of passing across functions?

def fix_random_seed(seed):
    import random
    import torch
    import numpy

    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
