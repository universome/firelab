import os
import sys
import random
import importlib.util
from itertools import product
from typing import List, Iterable

import yaml
import numpy
import torch
import torch.multiprocessing as mp

from .config import Config
from .utils.fs_utils import clean_dir, clean_file, touch_file, load_config
from .utils.training_utils import fix_random_seed, run_tensorboard
from .base_trainer import BaseTrainer


# TODO: move error msgs into separate file?
PATH_EXISTS_ERROR_MSG = ("`{}` directory or file already exists: "
    "have you already run this experiment? "
    "Provide `--overwrite` option if you want to overwrite the results.")
PATH_NOT_EXISTS_ERROR_MSG = ("`{}` directory or file does not exist")


def run(cmd:str, args):
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
    TrainerClass = getattr(trainers, config.get('trainer'))

    if config.has('hpo'):
        if config.firelab.has('continue_from_iter'):
            raise NotImplementedError("Continuing HPO is not yet implemented")

        run_hpo(TrainerClass, config)
    else:
        trainer = TrainerClass(config)
        trainer.start()


def run_hpo(TrainerClass, global_config):
    configs = spawn_config_for_hpo(global_config)

    clean_dir(os.path.join(global_config.firelab.experiments_dir, 'summaries'), create=True)

    # TODO: This is unacceptable! Care for situations,
    # where we have more GPUs than CPUs and
    # when we use several GPUs per experiment
    config_groups = group_experiments_by_gpus_used(configs)
    print('Will be using for HPO %d' % len(config_groups), 'processes')

    processes = []

    for group in config_groups:
        process = mp.spawn(hpo_series_runner, args=[TrainerClass, group], join=False)
        processes.append(process)

    for process in processes:
        process.join()


def hpo_series_runner(process_index:int, TrainerClass:BaseTrainer, configs_group:List[Config]):
    for config in configs_group:
        clean_dir(config.firelab.checkpoints_path, create=True)
        clean_dir(config.firelab.logs_path, create=True)

        trainer = TrainerClass(config)
        trainer.start()


def group_experiments_by_gpus_used(configs):
    gpus_to_group = {}

    for config in configs:
        gpus = tuple(sorted(config.available_gpus))

        if not gpus in gpus_to_group:
            gpus_to_group[gpus] = []

        gpus_to_group[gpus].append(config)

    return list(gpus_to_group.values())


def spawn_config_for_hpo(config):
    assert config.has('hpo')

    if not config.hpo.has('scheme'):
        print('Scheme for HPO is not specified. Gonna use grid search')
        config.hpo.set('scheme', 'grid-search')

    if config.hpo.scheme == 'grid-search':
        return spawn_config_for_grid_search_hpo(config)
    else:
        raise NotImplementedError # TODO


def spawn_config_for_grid_search_hpo(config) -> List[Config]:
    configs = []
    grid_dim_sizes = [len(config.hpo.grid.get(p)) for p in config.hpo.grid.keys()]
    vals_idx = [list(range(n)) for n in grid_dim_sizes]
    idx_list = list(product(*vals_idx))
    gpus_distribution = distribute_gpus_for_hpo(len(idx_list), config)

    for i, idx in enumerate(idx_list):
        values = [config.hpo.grid.get(p)[i] for p, i in zip(config.hpo.grid.keys(), idx)]
        new_config = config.to_dict()
        new_config.pop('hpo')

        for key, value in zip(config.hpo.grid.keys(), values):
            new_config['hp'][key] = value

        new_config['available_gpus'] = gpus_distribution[i]
        new_config['firelab']['checkpoints_path'] = os.path.join(new_config['firelab']['checkpoints_path'], 'hpo-experiment-%d' % i)
        new_config['firelab']['logs_path'] = os.path.join(new_config['firelab']['logs_path'], 'hpo-experiment-%d' % i)
        new_config['firelab']['summary_path'] = os.path.join(new_config['firelab']['experiments_dir'], 'summaries/hpo-experiment-%d.md' % i)

        configs.append(Config(new_config))

    return configs


def distribute_gpus_for_hpo(num_experiments:int, config:Config):
    num_gpus_per_experiment = config.hpo.get('num_gpus_per_experiment', 1)
    available_gpus = config.available_gpus

    assert len(available_gpus) >= num_gpus_per_experiment, """
        Amount of GPUs you want to allocate for each experiment is not available"""

    if num_gpus_per_experiment > 1:
        raise NotImplementedError # TODO

    distribution = (available_gpus * num_experiments)[:num_experiments]
    distribution = [[gpu] for gpu in distribution]

    return distribution


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
    config.firelab.set('reset_iters_counter', args.reset_iters_counter)

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

    # TODO: Validate all config properties
    assert not config.has('device'), '''You cannot set `firelab` manually.
                                        It's internally managed by FireLab framework.'''

    # Let's augment config with some helping stuff
    config.set('firelab', {
        'project_path': os.getcwd(),
        'experiments_dir': paths['experiments_dir'],
        'name': exp_name,
        'logs_path': paths['logs'],
        'checkpoints_path': paths['checkpoints'],
        'summary_path': paths['summary'],
    })

    if config.has('random_seed'):
        fix_random_seed(config.random_seed)
    else:
        print('Warn: random seed is not set. Consider setting it for reproducibility.')

    assert not config.has('device'), '''You cannot set `device` manually.
                                        Manage available GPUs via `available_gpus` parameter.'''


    # Setting available GPUs and proper device
    visible_gpus = list(range(torch.cuda.device_count()))

    if not config.has('available_gpus'):
        if len(visible_gpus) > 0:
            print('Found %d GPUs, but `available_gpus` parameter is not set. '\
                  'I gonna use them all!' % len(visible_gpus))

        config.set('available_gpus', visible_gpus)

    if len(config.available_gpus) > 0:
        config.set('device', 'cuda:%d' % config.available_gpus[0])
    else:
        config.set('device', 'cpu')

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
