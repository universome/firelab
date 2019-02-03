from firelab.manager import distribute_gpus_for_hpo
from firelab.config import Config


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
