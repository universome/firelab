import os
import sys
import random
import signal
import shutil
import importlib.util
from itertools import product
from typing import List, Iterable
from datetime import datetime

import yaml
import numpy
import torch
import torch.multiprocessing as mp

from .config import Config
from .utils.fs_utils import clean_dir, clean_file, touch_file, load_config
from .utils.training_utils import fix_random_seed, run_tensorboard
from .base_trainer import BaseTrainer


# TODO: move error msgs into separate file?
PATH_NOT_EXISTS_ERROR_MSG = ("`{}` directory or file does not exist")


def run(cmd:str, args):
    if cmd == 'start':
        config = create_new_experiment(args)
        start_experiment(config, tb_port=args.tb_port, stay_after_training=args.stay_after_training)
    elif cmd == 'continue':
        #continue_experiment(init_config(args), args)
        raise NotImplementedError
    elif cmd == 'tb':
        run_tensorboard_for_exp(args)
    elif cmd == 'ls':
        raise NotImplementedError
    elif cmd == 'clean':
        clean_experiments_by_prefix(args.prefix)
    else:
        raise NotImplementedError


def create_new_experiment(args):
    # TODO: looks like this lines should not be here
    config_name = os.path.basename(args.config_path)[:-4]
    exp_name = config_name + datetime.now().strftime('-%Y-%m-%d_%H-%M-%S')
    config = init_config(args.config_path, exp_name)

    os.makedirs(config.firelab.logs_path)
    os.makedirs(config.firelab.checkpoints_path)
    shutil.copyfile(args.config_path, config.firelab.config_path)

    print('New experiment created at:', os.path.join(config.firelab.experiments_dir, exp_name))

    return config


def start_experiment(config, tb_port:int=None, stay_after_training:bool=False):
    # TODO: ensure write access to the directory
    if not config.firelab.get('continue_from_iter') is None:
        validate_path_existence(config.firelab.logs_path, True)
        validate_path_existence(config.firelab.checkpoints_path, True)
        # validate_path_existence(config.firelab.summary_path, True) # TODO

    if config.firelab.get('continue_from_iter') is None:
        clean_dir(config.firelab.checkpoints_path, create=True)
        clean_dir(config.firelab.logs_path, create=True)

    if tb_port:
        print('Starting tensorboard on port', tb_port)
        run_tensorboard(config.firelab.logs_path, tb_port)

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

    if stay_after_training:
        print('Training was finished, but I gonna stay hanging here (because stay_after_training is enabled).')
        signal.pause()


def run_hpo(TrainerClass, global_config):
    # TODO: actually, we should use GPUs as they are getting free
    # Fixing all GPUs at once for each experiment is wrong,
    # because some experiments run MUCH faster than others

    configs = spawn_configs_for_hpo(global_config)

    clean_dir(os.path.join(global_config.firelab.experiments_dir,
                           global_config.firelab.exp_name, 'summaries'), create=True)

    # TODO: Is it ok to assume that we always have more CPUs than concurrent experiments?
    config_groups = group_experiments_by_gpus_used(configs)
    print('Num concurrent experiments to run: %d' % len(config_groups))

    processes = []
    n_parallel_per_gpu:int = global_config.hpo.get('num_parallel_experimens_per_gpu', 1)

    for group in config_groups:
        process = mp.spawn(hpo_series_runner, args=[TrainerClass, group, n_parallel_per_gpu], join=False)
        processes.append(process)

    for i, process in enumerate(processes):
        process.join()
        print('HPO series %d finished!' % (i + 1))


def hpo_series_runner(series_index:int, TrainerClass:BaseTrainer, configs_group:List[Config], n_parallel:int=1):
    parallel_groups = [configs_group[i:i+n_parallel] for i in range(0, len(configs_group), n_parallel)]

    for group in parallel_groups:
        parallel_processes = []

        for config in group:
            process = mp.spawn(run_single_hpo_experiment, args=[TrainerClass, config], join=False)
            parallel_processes.append(process)

        for i, process in enumerate(parallel_processes):
            process.join()
            print('HPO experiment #{} in series #{} finished!'.format(i, series_index))


def run_single_hpo_experiment(exp_idx:int, TrainerClass:BaseTrainer, config:Config):
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


def spawn_configs_for_hpo(config):
    assert config.has('hpo')

    if not config.hpo.has('scheme'):
        print('Scheme for HPO is not specified. Gonna use grid search')
        config.hpo.set('scheme', 'grid-search')

    if config.hpo.scheme == 'grid-search':
        return spawn_configs_for_grid_search_hpo(config)
    if config.hpo.scheme == 'random-search':
        return spawn_configs_for_random_search_hpo(config)
    else:
        raise NotImplementedError # TODO


def spawn_configs_for_grid_search_hpo(config) -> List[Config]:
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
        new_config['firelab']['device_name'] = 'cuda:%d' % gpus_distribution[i][0]
        new_config['firelab']['checkpoints_path'] = os.path.join(new_config['firelab']['checkpoints_path'], 'hpo-experiment-%d' % i)
        new_config['firelab']['logs_path'] = os.path.join(new_config['firelab']['logs_path'], 'hpo-experiment-%d' % i)
        new_config['firelab']['summary_path'] = os.path.join(new_config['firelab']['experiments_dir'], 'summaries/hpo-experiment-%d.yml' % i)
        new_config['firelab']['exp_name'] = '{}_hpo-experiment-{}'.format(new_config['firelab']['exp_name'], i)
        new_config['firelab']['config_path'] = os.path.join(new_config['firelab']['experiments_dir'], 'configs/hpo-experiment-%d.yml' % i)

        configs.append(Config(new_config))

    return configs


def spawn_configs_for_random_search_hpo(config:Config) -> List[Config]:
    configs = spawn_configs_for_grid_search_hpo(config)
    configs = random.sample(configs, config.hpo.num_experiments)

    return configs


def distribute_gpus_for_hpo(num_experiments:int, config:Config) -> List[List[int]]:
    num_gpus_per_experiment = config.hpo.get('num_gpus_per_experiment', 1)
    available_gpus = config.available_gpus

    assert len(available_gpus) >= num_gpus_per_experiment, """
        Amount of GPUs you want to allocate for each experiment is not available"""

    num_unused_gpus = len(available_gpus) % num_gpus_per_experiment
    num_concurrent_experiments = (len(available_gpus) - num_unused_gpus) // num_gpus_per_experiment
    gpu_groups_idx = [list(range(
        num_gpus_per_experiment * i, num_gpus_per_experiment * i + num_gpus_per_experiment
    )) for i in range(num_concurrent_experiments)]
    gpu_groups = [[available_gpus[gpu_i] for gpu_i in group_idx] for group_idx in gpu_groups_idx]
    distribution = [gpu_groups[i % len(gpu_groups)] for i in range(num_experiments)]

    if num_unused_gpus != 0:
        print('You specified {} GPUs per experiment and {} GPUs are available. ' \
              'So {} GPUs will be unused :('.format(
                  num_gpus_per_experiment, len(available_gpus), num_unused_gpus))

    return distribution


# def continue_experiment(config, args):
#     # Finding latest checkpoint
#     checkpoints = os.listdir(config.firelab.checkpoints_path)

#     if checkpoints == []:
#         raise Exception('Can\'t continue: no checkpoints are available')

#     if args.iteration is None:
#         iters = [int(c.split('.')[0].split('-')[-1]) for c in checkpoints]
#         iteration = max(iters)
#     else:
#         iteration = args.iteration

#     print('Continuing from iteration #{}.'.format(iteration))
#     config.firelab.set('continue_from_iter', iteration)
#     config.firelab.set('reset_iters_counter', args.reset_iters_counter)

#     start_experiment(config, args)


def init_config(config_path:str, exp_name:str):
    paths = compute_paths(exp_name)
    config = load_config(config_path)

    # TODO: Validate all config properties
    assert not config.has('firelab'), \
        'You cannot set `firelab` manually. It is internally managed by FireLab framework.'

    # Let's augment config with some helping stuff
    config.set('firelab', {
        'config_path': paths['config'],
        'project_path': os.getcwd(),
        'experiments_dir': paths['experiments_dir'],
        'exp_name': exp_name,
        'logs_path': paths['logs'],
        'checkpoints_path': paths['checkpoints'],
        'summary_path': paths['summary'],
    })

    if config.has('random_seed'):
        fix_random_seed(config.random_seed)
    else:
        print('Warn: random seed is not set. Consider setting it for reproducibility.')

    # Setting available GPUs and proper device
    visible_gpus = list(range(torch.cuda.device_count()))

    if not config.has('available_gpus'):
        if len(visible_gpus) > 0:
            print('Found %d GPUs, but `available_gpus` parameter is not set. '\
                  'I gonna use them all!' % len(visible_gpus))

        config.set('available_gpus', visible_gpus)

    assert not config.has('device_name'), \
        'FireLab detects and sets device_name for you. You influence it via `available_gpus`.'

    if len(config.available_gpus) > 0:
        config.firelab.set('device_name', 'cuda:%d' % config.available_gpus[0])
    else:
        config.firelab.set('device_name', 'cpu')

    # TODO: make config immutable

    return config


def compute_paths(exp_name):
    "Calculates paths for a given experiment"
    experiments_dir = get_experiments_dir()

    return {
        'experiments_dir': experiments_dir,
        'config': os.path.join(experiments_dir, exp_name, "config.yml"),
        'logs': os.path.join(experiments_dir, exp_name, "logs"),
        'checkpoints': os.path.join(experiments_dir, exp_name, "checkpoints"),
        'summary': os.path.join(experiments_dir, exp_name, "summary.yml"),
    }

def get_experiments_dir():
    # TODO: We can't rely on os.getcwd(). How to get project dir properly?
    return os.path.join(os.getcwd(), "experiments")


def validate_path_existence(path, should_exist):
    if should_exist and not os.path.exists(path):
        raise Exception(PATH_NOT_EXISTS_ERROR_MSG.format(path))

    if not should_exist and os.path.exists(path):
        raise Exception(PATH_EXISTS_ERROR_MSG.format(path))


def run_tensorboard_for_exp(args):
    config_path = compute_paths(args.exp_name)['config']
    config = init_config(config_path, args.exp_name)
    run_tensorboard(config.firelab.logs_path, args.tb_port)
    signal.pause()


def clean_experiments_by_prefix(prefix:str):
    experiments_dir = get_experiments_dir()

    for dir in os.listdir(experiments_dir):
        if dir.startswith(prefix):
            dir_path = os.path.join(experiments_dir, dir)
            print('Removing', dir_path)
            shutil.rmtree(dir_path)

    print('Done')
