import os
import sys
import random
import signal
import shutil
import logging
import traceback
import importlib.util
from concurrent.futures import ProcessPoolExecutor, wait
from torch.multiprocessing import Manager, Lock
from typing import List, Iterable, Tuple

import numpy
import torch
import coloredlogs

from .config import Config
from .utils.fs_utils import clean_dir, clean_file, touch_file, load_config, check_that_path_exists, infer_new_experiment_version
from .utils.training_utils import fix_random_seed, run_tensorboard
from .base_trainer import BaseTrainer
from .hpo import spawn_configs_for_hpo

torch.multiprocessing.set_start_method("spawn", force=True)
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


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
    version = infer_new_experiment_version(get_experiments_dir(), config_name)
    exp_name = f'{config_name}-{version:05d}'
    config = init_config(args.config_path, exp_name)

    # TODO: Trainer should do this thing, no?
    # shutil.copyfile(args.config_path, config.firelab.paths.config_path)

    return config


def start_experiment(config, tb_port:int=None, stay_after_training:bool=False):
    # TODO: ensure write access to the directory
    if config.firelab.has('continue_from_iter'):
        check_that_path_exists(config.firelab.paths.logs_path)
        check_that_path_exists(config.firelab.paths.checkpoints_path)
        # check_that_path_exists(config.firelab.summary_path) # TODO

    if tb_port:
        logger.info(f'Starting tensorboard on port {tb_port}')
        run_tensorboard(config.firelab.paths.logs_path, tb_port)

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
        logger.info('Training was finished, but I gonna stay hanging here (because stay_after_training is enabled).')
        signal.pause()


def run_hpo(TrainerClass, global_config):
    # TODO: Is it ok to assume that we always have more CPUs than concurrent experiments?
    configs = spawn_configs_for_hpo(global_config)
    clean_dir(os.path.join(global_config.firelab.paths.experiments_dir,
                           global_config.firelab.exp_name, 'summaries'), create=True)

    n_parallel = global_config.hpo.get('num_parallel_experiments_per_gpu', 1) \
               * (len(global_config.firelab.gpus) \
               // global_config.hpo.get('num_gpus_per_experiment', 1))

    logger.info(f'Total number of experiments to run: {len(configs)}')
    logger.info(f'Num concurrent HPO experiments to run: {n_parallel}')

    gpus:Tuple[int] = tuple(global_config.firelab.gpus)
    gpus_usage = Manager().list([0] * len(gpus))
    gpus_usage_lock = Manager().Lock()
    futures = []

    with ProcessPoolExecutor(n_parallel) as executor:
        for config in configs:
            args = [
                TrainerClass,
                config,
                global_config.hpo.get('num_gpus_per_experiment', 1),
                global_config.hpo.get('num_parallel_experiments_per_gpu', 1),
                gpus_usage,
                gpus_usage_lock,
                gpus
            ]

            future = executor.submit(run_single_hpo_experiment, *args)
            futures.append(future)

        wait(futures)


def run_single_hpo_experiment(TrainerClass:BaseTrainer,
                              config:Config,
                              n_gpus_required:int,
                              n_experiments_per_gpu:int,
                              gpus_usage:Manager,
                              gpus_usage_lock:Lock,
                              gpus:Tuple[int]):
    try:
        with gpus_usage_lock:
            free_gpus_idx = [i for i, _ in enumerate(gpus) if gpus_usage[i] < n_experiments_per_gpu]
            gpus_idx_to_take = free_gpus_idx[:n_gpus_required]
            gpus_to_take = [gpus[i] for i in gpus_idx_to_take]

            logger.info(f'[{config.firelab.exp_name}] GPUs usage: {gpus_usage}. GPUs to take: {gpus_to_take}.')

            # Taking GPUs
            for gpu_idx in gpus_idx_to_take:
                gpus_usage[gpu_idx] += 1

        config.firelab.set('gpus', gpus_to_take)
        config.firelab.set('device_name', f'cuda:{gpus_to_take[0]}')
        config.save(config.firelab.paths.config_path) # Saving config for future

        trainer = TrainerClass(config)
        trainer.start()
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        logger.error(f"{e}")
        raise
    finally:
        # Releasing GPUs
        with gpus_usage_lock:
            for gpu_idx in gpus_idx_to_take:
                gpus_usage[gpu_idx] -= 1


# def continue_experiment(config, args):
#     # Finding latest checkpoint
#     checkpoints = os.listdir(config.firelab.paths.checkpoints_path)

#     if checkpoints == []:
#         raise Exception('Can\'t continue: no checkpoints are available')

#     if args.iteration is None:
#         iters = [int(c.split('.')[0].split('-')[-1]) for c in checkpoints]
#         iteration = max(iters)
#     else:
#         iteration = args.iteration

#     logger.info('Continuing from iteration #{}.'.format(iteration))
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
    # TODO: assign paths automatically, because we have a lot of duplication
    config.set('firelab', {
        'exp_name': exp_name,
        'paths': {
            'config_path': paths['config_path'],
            'project_path': os.getcwd(),
            'experiments_dir': paths['experiments_dir'],
            'logs_path': paths['logs_path'],
            'checkpoints_path': paths['checkpoints_path'],
            'summary_path': paths['summary_path'],
            'custom_data_path': paths['custom_data_path']
        }
    })

    if config.has('random_seed'):
        fix_random_seed(config.random_seed)
    else:
        logger.info('Warn: random seed is not set. Consider setting it for reproducibility.')

    # Setting available GPUs and proper device
    visible_gpus = list(range(torch.cuda.device_count()))

    if not config.has('gpus'):
        if len(visible_gpus) > 0:
            logger.info(f'Found {len(visible_gpus)} GPUs, but `gpus` parameter is not set. '\
                  'I gonna use them all!')

        config.firelab.set('gpus', visible_gpus)
    else:
        config.firelab.set('gpus', [gpu for gpu in config.gpus])

    # if len(config.firelab.gpus) > 0:
    #     config.firelab.set('device_name', 'cuda:%d' % config.firelab.gpus[0])
    # else:
    #     config.firelab.set('device_name', 'cpu')

    # TODO: make config immutable

    return config


def compute_paths(exp_name):
    "Calculates paths for a given experiment"
    experiments_dir = get_experiments_dir()

    return {
        'experiments_dir': experiments_dir,
        'config_path': os.path.join(experiments_dir, exp_name, "config.yml"),
        'logs_path': os.path.join(experiments_dir, exp_name, "logs"),
        'checkpoints_path': os.path.join(experiments_dir, exp_name, "checkpoints"),
        'summary_path': os.path.join(experiments_dir, exp_name, "summary.yml"),
        'custom_data_path': os.path.join(experiments_dir, exp_name, 'custom_data')
    }


def get_experiments_dir():
    # TODO: We can't rely on os.getcwd(). How to get project dir properly?
    return os.path.join(os.getcwd(), "experiments")


def run_tensorboard_for_exp(args):
    config_path = compute_paths(args.exp_name)['config_path']
    config = init_config(config_path, args.exp_name)
    run_tensorboard(config.firelab.paths.logs_path, args.tb_port)
    signal.pause()


def clean_experiments_by_prefix(prefix:str):
    experiments_dir = get_experiments_dir()

    for dir in os.listdir(experiments_dir):
        if dir.startswith(prefix):
            dir_path = os.path.join(experiments_dir, dir)
            logger.info(f'Removing {dir_path}')
            shutil.rmtree(dir_path)

    logger.info('Done')
