import os
import random
import logging
from typing import List
from itertools import product

import coloredlogs

from .config import Config


logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


def spawn_configs_for_hpo(config):
    assert config.has('hpo')

    if not config.hpo.has('scheme'):
        logger.info('Scheme for HPO is not specified. Gonna use grid search')
        config.hpo.set('scheme', 'grid-search')

    if config.hpo.scheme == 'grid-search':
        return spawn_configs_for_grid_search_hpo(config)
    if config.hpo.scheme == 'random-search':
        return spawn_configs_for_random_search_hpo(config)
    else:
        raise NotImplementedError # TODO


def spawn_configs_for_grid_search_hpo(config) -> List[Config]:
    experiments_vals_idx = compute_hpo_vals_idx(config.hpo.grid)
    gpus_distribution = distribute_gpus_for_hpo(len(experiments_vals_idx), config)
    configs = create_hpo_configs(config, experiments_vals_idx, gpus_distribution)

    return configs


def spawn_configs_for_random_search_hpo(config:Config) -> List[Config]:
    experiments_vals_idx = random.sample(compute_hpo_vals_idx(config.hpo.grid), config.hpo.num_experiments)
    gpus_distribution = distribute_gpus_for_hpo(len(experiments_vals_idx), config)
    configs = create_hpo_configs(config, experiments_vals_idx, gpus_distribution)

    return configs


def create_hpo_configs(config:Config, idx_list:List[List[int]], gpus_distribution:List[List[int]]) -> List[Config]:
    configs = []

    for i, idx in enumerate(idx_list):
        values = [config.hpo.grid.get(p)[i] for p, i in zip(config.hpo.grid.keys(), idx)]
        new_config = config.to_dict()
        new_config.pop('hpo')

        for key, value in zip(config.hpo.grid.keys(), values):
            new_config['hp'][key] = value

        new_config['available_gpus'] = gpus_distribution[i]
        new_config['firelab']['device_name'] = f'cuda:{gpus_distribution[i][0]}'
        new_config['firelab']['checkpoints_path'] = os.path.join(new_config['firelab']['checkpoints_path'], f'hpo-experiment-{i:03d}')
        new_config['firelab']['logs_path'] = os.path.join(new_config['firelab']['logs_path'], f'hpo-experiment-{i}')
        new_config['firelab']['summary_path'] = os.path.join(new_config['firelab']['experiments_dir'], new_config['firelab']['exp_name'], f'summaries/hpo-experiment-{i:03d}.yml')
        new_config['firelab']['exp_name'] = f"{new_config['firelab']['exp_name']}_hpo-experiment-{i:03d}"
        new_config['firelab']['config_path'] = os.path.join(new_config['firelab']['experiments_dir'], new_config['firelab']['exp_name'], f'configs/hpo-experiment-{i:03d}.yml')

        configs.append(Config(new_config))

    return configs


def compute_hpo_vals_idx(hpo_grid:Config) -> List[List[int]]:
    grid_dim_sizes = [len(hpo_grid.get(p)) for p in hpo_grid.keys()]
    vals_idx = [list(range(n)) for n in grid_dim_sizes]
    experiments_vals_idx = list(product(*vals_idx))

    return experiments_vals_idx


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
        logger.info(f'You specified {num_gpus_per_experiment} GPUs per experiment ' \
             f'and {len(available_gpus)} GPUs are available. So {num_unused_gpus} GPUs will be unused :(')

    return distribution
