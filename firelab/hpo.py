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
        raise NotImplementedError(f'HPO scheme {config.hpo.scheme} is not supported. Use grid-search or random-search, please.')


def spawn_configs_for_grid_search_hpo(config) -> List[Config]:
    experiments_vals_idx = compute_hpo_vals_idx(config.hpo.grid)
    configs = create_hpo_configs(config, experiments_vals_idx)

    return configs


def spawn_configs_for_random_search_hpo(config:Config) -> List[Config]:
    experiments_vals_idx = random.sample(compute_hpo_vals_idx(config.hpo.grid), config.hpo.num_experiments)
    configs = create_hpo_configs(config, experiments_vals_idx)

    return configs


def create_hpo_configs(config:Config, idx_list:List[List[int]]) -> List[Config]:
    configs = []

    for i, idx in enumerate(idx_list):
        values = [config.hpo.grid.get(p)[i] for p, i in zip(config.hpo.grid.keys(), idx)]
        new_config = config.to_dict()
        new_config.pop('hpo')
        new_config['firelab'].pop('available_gpus')
        new_config['firelab'].pop('device_name')

        for key, value in zip(config.hpo.grid.keys(), values):
            new_config['hp'][key] = value

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
