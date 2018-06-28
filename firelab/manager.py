import os
import sys
import random
import importlib.util

import yaml
import numpy
import torch

from .utils import clean_dir, fix_random_seed


# TODO: move error msgs into separate file?
PATH_EXISTS_ERROR_MSG = ("`{}` directory or file already exists: "
    "have you already run this experiment? "
    "Provide `--overwrite` option if you want to overwrite the results.")
PATH_NOT_EXISTS_ERROR_MSG = ("`{}` directory or file does not exist")


def run(cmd, args):
    config = load_config(args)

    if cmd == 'start':
        start_experiment(config, args)
    elif cmd == 'continue':
        continue_experiment(config, args)
    elif cmd == 'ls':
        raise NotImplementedError
    else:
        raise NotImplementedError


def start_experiment(config, args):
    if 'continue_from_iter' in config['firelab']:
        validate_path_existence(config['firelab']['logs_path'], True)
        validate_path_existence(config['firelab']['checkpoints_path'], True)
        # validate_path_existence(config['firelab']['summary_path'], True)
    elif args.overwrite is False:
        validate_path_existence(config['firelab']['logs_path'], False)
        validate_path_existence(config['firelab']['checkpoints_path'], False)
        validate_path_existence(config['firelab']['summary_path'], False)


    # TODO: ensure write access to the directory
    if not 'continue_from_iter' in config['firelab']:
        clean_dir(config['checkpoints_path'])
        clean_dir(config['logs_path'])

    # TODO: are there any better ways to reach src.trainers?
    sys.path.append(os.getcwd())
    trainers = importlib.import_module('src.trainers')
    trainer_cls = getattr(trainers, config.get('trainer'))
    trainer = trainer_cls(config)
    trainer.start()


def continue_experiment(config, args):
    # Finding latest checkpoint
    checkpoints = os.listdir(config['firelab']['checkpoints_path'])

    if checkpoints == []:
        raise Exception('Can\'t continue: no checkpoints are available')

    iters = [int(c[:-4].split('-')[-1]) for c in checkpoints]
    latest_iter = max(iters)

    print('Latest checkpoint found: {}. Continuing from it.'.format(latest_iter))
    config['firelab']['continue_from_iter'] = latest_iter
    start_experiment(config, args)


def load_config(args):
    # TODO: We can't rely on os.getcwd(). How to get project dir properly?
    experiments_dir = os.path.join(os.getcwd(), "experiments")
    exp_name = args.name # Name of the experiment is the same as config name
    config_path = os.path.join(experiments_dir, exp_name, "config.yml")
    logs_path = os.path.join(experiments_dir, exp_name, "logs")
    checkpoints_path = os.path.join(experiments_dir, exp_name, "checkpoints")
    summary_path = os.path.join(experiments_dir, exp_name, "summary.md")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

        # TODO: validate config
        assert not 'firelab' in config

        # Let's augment config with some helping stuff
        config['firelab'] = {}
        config['firelab']['project_path'] = os.getcwd()
        config['firelab']['name'] = exp_name
        config['firelab']['logs_path'] = logs_path
        config['firelab']['checkpoints_path'] = checkpoints_path
        config['firelab']['experiments_dir'] = experiments_dir
        config['firelab']['summary_path'] = summary_path

        if 'random_seed' in config:
            fix_random_seed(config['random_seed'])

        # TODO: make config immutable

    return config


def validate_path_existence(path, should_exist):
    if should_exist and not os.path.exists(path):
        raise Exception(PATH_NOT_EXISTS_ERROR_MSG.format(path))

    if not should_exist and os.path.exists(path):
        raise Exception(PATH_EXISTS_ERROR_MSG.format(path))
