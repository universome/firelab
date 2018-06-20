import os
from collections import namedtuple
import shutil

import torch

# TODO: check `no_cuda` argument in config
use_cuda = torch.cuda.is_available()
HPLinearScheme = namedtuple('HPLinearScheme', ['start_val', 'end_val', 'period'])

def cudable(x):
    """
    Transforms torch tensor/module to cuda tensor/module
    """
    return x.cuda() if use_cuda and is_cudable(x) else x


def is_cudable(x):
    return isinstance(x, torch.Tensor) or isinstance(x, torch.nn.Module)


def compute_param_by_scheme(scheme:HPLinearScheme, num_iters_done:int):
    """
    :param scheme: we are increasing param from scheme.start_val to scheme.end_val
                   during scheme.period
    :param num_iters_done
    """
    t1, t2, period = scheme

    if num_iters_done >= period:
        return t2
    else:
        if t1 > t2:
            return t1 - (t1 - t2) * num_iters_done / period
        else:
            return t1 + (t2 - t1) * num_iters_done / period


def clean_dir(dir, create=True):
    """Deletes everything inside directory. Creates it if does not exist"""
    # TODO: Can't tensorboard use only latest log?
    if os.path.exists(dir): shutil.rmtree(dir)
    os.mkdir(dir)

