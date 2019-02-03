from firelab.config import Config
from firelab.manager import distribute_gpus_for_hpo, group_experiments_by_gpus_used


def test_configs_grouping():
    assert len(group_experiments_by_gpus_used([ \
        Config({'available_gpus': [0,1]}), \
        Config({'available_gpus': [1,2]}), \
        Config({'available_gpus': [0,1]}), \
        Config({'available_gpus': [1,2]}), \
    ])) == 2

    assert len(group_experiments_by_gpus_used([ \
        Config({'available_gpus': [0,1]}), \
        Config({'available_gpus': [2,3]}), \
        Config({'available_gpus': [4,5]}), \
        Config({'available_gpus': [3,2]}), \
    ])) == 3


def test_gpus_distribution():
    assert distribute_gpus_for_hpo(4, Config({'available_gpus': [0,1,2,3], 'hpo': {}})) \
        == [[0], [1], [2], [3]]

    assert distribute_gpus_for_hpo(4, Config({'available_gpus': [3,7,11,2], 'hpo': {}})) \
        == [[3], [7], [11], [2]]

    assert distribute_gpus_for_hpo(4, Config({'available_gpus': [0,1,2,3], 'hpo': {'num_gpus_per_experiment': 2}})) \
        == [[0,1], [2,3], [0,1], [2,3]]

    assert distribute_gpus_for_hpo(10, Config({'available_gpus': [0,1,2,3], 'hpo': {}})) \
        == [[0],[1],[2],[3],[0],[1],[2],[3],[0],[1]]

    assert distribute_gpus_for_hpo(10, Config({'available_gpus': [0,1,2,3], 'hpo': {'num_gpus_per_experiment': 2}})) \
        == [[0,1], [2,3]] * 5

    assert distribute_gpus_for_hpo(10, Config({'available_gpus': [0,1,2,3], 'hpo': {'num_gpus_per_experiment': 3}})) \
        == [[0,1,2]] * 10
