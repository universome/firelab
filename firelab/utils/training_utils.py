import os
import gc
import subprocess
import atexit

import torch
import numpy as np


# TODO: check `no_cuda` argument in config
use_cuda = torch.cuda.is_available()


def cudable(x):
    "Transforms torch tensor/module to cuda tensor/module"
    if not use_cuda: return x

    if hasattr(x, "cuda") and callable(getattr(x, "cuda")):
        return x.to('cuda')

    # Couldn't transfer it on GPU (or it is not transferable).
    return x


class LinearScheme:
    """
    We are increasing param from `start_val` to `end_val` during `period`
    """
    def __init__(self, start_val, end_val, period):
        self.start_val = start_val
        self.end_val = end_val
        self.period = period

    def evaluate(self, iteration):
        if iteration >= self.period:
            return self.end_val
        else:
            return self.start_val + (self.end_val - self.start_val) * iteration / self.period


class PiecewiseLinearScheme:
    "We apply LinearScheme for each piece"
    def __init__(self, pieces):
        self.schemes = [LinearScheme(*p) for p in pieces]

    def scheme_idx_for_iteration(self, iteration):
        curr_sum = 0

        for i in range(len(self.schemes) - 1):
            if curr_sum <= iteration <= (curr_sum + self.schemes[i].period):
                return i

            curr_sum += self.schemes[i].period

        return len(self.schemes) - 1

    def evaluate(self, iteration):
        scheme_idx = self.scheme_idx_for_iteration(iteration)
        num_iters_to_ignore = sum([s.period for s in self.schemes[:scheme_idx]])

        return self.schemes[scheme_idx].evaluate(iteration - num_iters_to_ignore)


def proportion_coef(x:float, y:float, proportion:float):
    "On what number should we multiply x to get (proportion * y)?"
    assert x >= 0
    assert y >= 0
    assert 0 <= proportion <= 1

    return y * proportion / x


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
        return np.argmin(history[-n_steps:]) != 0
    else:
        return np.argmax(history[-n_steps:]) != 0


def run_tensorboard(logdir, port):
    # TODO(universome): well, tensorboard has some kind of python API,
    # but we can't call tb using it, because there are some conflicting
    # proto files between tensorboardX and tensorboard
    # https://github.com/lanpa/tensorboardX/issues/206

    # Let's remove gpu from tensorboard, because it eats too much
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    proc = subprocess.Popen(['tensorboard', '--logdir', logdir, '--port', str(port)], env=env)
    atexit.register(lambda: proc.terminate())


def grad_norm(params_gen, p=2):
    """
    Computes norm of the gradient for a given parameters list
    """
    return sum([w.grad.norm(p) ** p for w in params_gen]) ** (1 / p)


def check_if_oom(e: Exception):
    """Checks if the given exception is cuda OOM"""
    # TODO: is  there a better way for this?
    return "out of memory" in str(e)


def safe_oom_call(fn, logger, *args, debug=False, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        if not check_if_oom(e): raise

        logger.error('Encountered CUDA OOM error in {}. Ignoring it.'.format(fn.__name__))
        torch.cuda.empty_cache()

        if debug:
            print_gpu_inhabitants(logger)


def print_gpu_inhabitants(logger):
    logger.info('Here are the guys who can live on GPU')

    for o in gc.get_objects():
        if torch.is_tensor(o):
            logger.info(o.type(), o.size())


def determine_turn(iteration:int, sequencing:list):
    """
    Determines turn for a given sequencing based on current iteration
    Assumes that iterations starts with 0
    Useful for turn-based training (for example, GANs)
    """
    # We do not care about the past
    iteration = iteration % sum(sequencing)

    # Let's find whose turn is now
    for i in range(len(sequencing)):
        if sum(sequencing[:i]) <= iteration < sum(sequencing[:i+1]):
            return i

    assert False, "Impossible scenario in determine_turn"


def get_module_device(m):
    try:
        return next(iter(m.parameters())).device
    except StopIteration:
        # TODO: maybe we should fall with error?
        return torch.device('cpu')
