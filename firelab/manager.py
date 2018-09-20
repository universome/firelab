import os
import sys
import random
import importlib.util

import yaml
import numpy
import torch

from .utils.fs_utils import clean_dir, clean_file, touch_file, load_config
from .utils.training_utils import fix_random_seed, run_tensorboard


# TODO: move error msgs into separate file?
PATH_EXISTS_ERROR_MSG = ("`{}` directory or file already exists: "
    "have you already run this experiment? "
    "Provide `--overwrite` option if you want to overwrite the results.")
PATH_NOT_EXISTS_ERROR_MSG = ("`{}` directory or file does not exist")


def run(cmd, args):
    if cmd == 'start':
        start_experiment(init_config(args), args)
    elif cmd == 'continue':
        continue_experiment(init_config(args), args)
    elif cmd == 'touch':
        create_blank_experiment(args)
    elif cmd == 'clean':
        clean_experiment(init_config(args), args)
    elif cmd == 'ls':
        raise NotImplementedError
    else:
        raise NotImplementedError


def start_experiment(config, args):
    # TODO: ensure write access to the directory

    if not config.firelab.get('continue_from_iter') is None:
        validate_path_existence(config.firelab.logs_path, True)
        validate_path_existence(config.firelab.checkpoints_path, True)
        # validate_path_existence(config.firelab.summary_path, True) # TODO
    elif args.overwrite is False:
        validate_path_existence(config.firelab.logs_path, False)
        validate_path_existence(config.firelab.checkpoints_path, False)
        validate_path_existence(config.firelab.summary_path, False)

    if config.firelab.get('continue_from_iter') is None:
        clean_dir(config.firelab.checkpoints_path, create=True)
        clean_dir(config.firelab.logs_path, create=True)

    if args.tb_port:
        print('Starting tensorboard on port', args.tb_port)
        run_tensorboard(config.firelab.logs_path, args.tb_port)

    # TODO: are there any better ways to reach src.trainers?
    sys.path.append(os.getcwd())
    trainers = importlib.import_module('src.trainers')
    trainer_cls = getattr(trainers, config.get('trainer'))
    trainer = trainer_cls(config)
    trainer.start()


def continue_experiment(config, args):
    # Finding latest checkpoint
    checkpoints = os.listdir(config.firelab.checkpoints_path)

    if checkpoints == []:
        raise Exception('Can\'t continue: no checkpoints are available')

    if args.iteration is None:
        iters = [int(c.split('.')[0].split('-')[-1]) for c in checkpoints]
        iteration = max(iters)
    else:
        iteration = args.iteration

    print('Continuing from iteration #{}.'.format(iteration))
    config.firelab.set('continue_from_iter', iteration)

    start_experiment(config, args)


def create_blank_experiment(args):
    exp_name = args.name
    paths = compute_paths(exp_name)
    exp_dir = os.path.join(paths['experiments_dir'], exp_name)
    validate_path_existence(exp_dir, False)

    os.mkdir(exp_dir)
    touch_file(paths['config'])
    os.mkdir(paths['logs'])
    os.mkdir(paths['checkpoints'])
    touch_file(paths['summary'])


def clean_experiment(config, args):
    "Removes logs/ and checkpoints/ content of the experiment"
    clean_dir(config.firelab.logs_path, create=True)
    clean_dir(config.firelab.checkpoints_path, create=True)
    clean_file(config.firelab.summary_path, create=True)


def init_config(args):
    exp_name = args.name # Name of the experiment is the same as config name
    paths = compute_paths(exp_name)

    if not os.path.isfile(paths['config']):
        raise FileNotFoundError(paths['config'])

    config = load_config(paths['config'])

    # TODO: validate config
    assert config.get('firelab') is None

    # Let's augment config with some helping stuff
    config.set('firelab', {
        'project_path': os.getcwd(),
        'experiments_dir': paths['experiments_dir'],
        'name': exp_name,
        'logs_path': paths['logs'],
        'checkpoints_path': paths['checkpoints'],
        'summary_path': paths['summary'],
    })

    if not config.get('random_seed') is None:
        fix_random_seed(config.random_seed)

    if config.get('hpo'):
        # Wow, this gonna be hot
        # We'll run several experiments on all available GPUs for HPO

        # Let's first generate configs for experiments

        # Now we are ready to run each experiment individually
        for gpu_idx in range(torch.cuda.device_count()):
            # Unfortunately, the only way to specify multiple GPUs
            # in pytorch is via CUDA_VISIBLE_DEVICES=...
            # print(gpu_idx)
            pass

    # TODO: make config immutable

    return config

def get_experiments_dir():
    # TODO: We can't rely on os.getcwd(). How to get project dir properly?
    return os.path.join(os.getcwd(), "experiments")


def compute_paths(exp_name):
    "Calculates paths for a given experiment"

    experiments_dir = get_experiments_dir()

    return {
        'experiments_dir': experiments_dir,
        'config': os.path.join(experiments_dir, exp_name, "config.yml"),
        'logs': os.path.join(experiments_dir, exp_name, "logs"),
        'checkpoints': os.path.join(experiments_dir, exp_name, "checkpoints"),
        'summary': os.path.join(experiments_dir, exp_name, "summary.md"),
    }


def validate_path_existence(path, should_exist):
    if should_exist and not os.path.exists(path):
        raise Exception(PATH_NOT_EXISTS_ERROR_MSG.format(path))

    if not should_exist and os.path.exists(path):
        raise Exception(PATH_EXISTS_ERROR_MSG.format(path))
