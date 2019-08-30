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
        new_config['firelab'].pop('gpus')
        new_config['hp'] = new_config.get('hp', {})

        for key, value in zip(config.hpo.grid.keys(), values):
            new_config['hp'][key] = value

        old_exp_name = new_config['firelab']['exp_name']
        new_config['firelab']['exp_name'] = f"{old_exp_name}_hpo-experiment-{i:05d}"
        new_config['firelab']['paths'] = {
            'project_path': new_config['firelab']['paths']['project_path'],
            'checkpoints_path': os.path.join(
                new_config['firelab']['paths']['experiments_dir'],
                old_exp_name,
                'checkpoints',
                f'hpo-experiment-{i:05d}'
            ),
            'logs_path': os.path.join(
                new_config['firelab']['paths']['experiments_dir'],
                old_exp_name,
                'logs',
                f'hpo-experiment-{i:05d}'
            ),
            'summary_path': os.path.join(
                new_config['firelab']['paths']['experiments_dir'],
                old_exp_name,
                'summaries',
                f'hpo-experiment-{i:05d}.yml'
            ),
            'config_path': os.path.join(
                new_config['firelab']['paths']['experiments_dir'],
                old_exp_name,
                'configs',
                f'hpo-experiment-{i:05d}.yml'
            ),
            'custom_data_path': os.path.join(
                new_config['firelab']['paths']['experiments_dir'],
                old_exp_name,
                'custom_data',
                f'hpo-experiment-{i:05d}'
            )
        }

        configs.append(Config(new_config))

    return configs


def compute_hpo_vals_idx(hpo_grid:Config) -> List[List[int]]:
    grid_dim_sizes = [len(hpo_grid.get(p)) for p in hpo_grid.keys()]
    vals_idx = [list(range(n)) for n in grid_dim_sizes]
    experiments_vals_idx = list(product(*vals_idx))

    return experiments_vals_idx
