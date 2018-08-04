import os
from collections import namedtuple
import shutil
import subprocess

import torch
import numpy as np

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
    start_val, end_val, period = scheme

    if num_iters_done >= period:
        return end_val
    else:
        return start_val + (end_val - start_val) * num_iters_done / period


def clean_dir(dir, create=True):
    """Deletes everything inside directory. Creates it if does not exist"""
    # TODO: Can't tensorboard use only latest log?
    if os.path.exists(dir): shutil.rmtree(dir)
    os.mkdir(dir)


def fix_random_seed(seed):
    import random
    import torch
    import numpy

    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def is_history_improving(history, n_steps: int, should_decrease: bool):
    """
    Checks by history if it has being improved since last 3 steps
    Returns true if the history is too short

    Arguments:
        - history: Iterable — history of the metric to check
        - n_steps: metric should have been improved at least once during last 3 steps
        - should_increase: flag which says if it should grow up or down
    """
    if len(history) < n_steps: return True

    if should_decrease:
        return np.argmin(history[-n_steps:]) == 0
    else:
        return np.argmax(history[-n_steps:]) == 0

def run_tensorboard(logdir, port):
    # TODO(universome): well, tensorboard has some kind of python API,
    # but we can't call tb using it, because there are some conflicting
    # proto files between tensorboardX and tensorboard
    # https://github.com/lanpa/tensorboardX/issues/206
    subprocess.Popen(['tensorboard', '--logdir', logdir, '--port', str(port)])
