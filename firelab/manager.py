import os
import sys
import importlib.util

import yaml
import numpy
import torch

from .utils import clean_dir

# TODO: move error msgs into separate file?
RESULTS_EXIST_ERROR_MSG = ("`{}` directory or file already exists: "
    "have you already run this experiment? "
    "Provide `--overwrite` option if you want to overwrite the results.")

def start_experiment(args):
    # TODO: We can't rely on os.getcwd(). How to get project dir properly?
    experiments_dir = os.path.join(os.getcwd(), "experiments")
    exp_name = args.name # Name of the experiment is the same as config name

    config_path = os.path.join(experiments_dir, exp_name, "config.yml")
    logs_path = os.path.join(experiments_dir, exp_name, "logs")
    checkpoints_path = os.path.join(experiments_dir, exp_name, "checkpoints")
    summary_path = os.path.join(experiments_dir, exp_name, "summary.md")

    if not os.path.isfile(config_path): raise FileNotFoundError(config_path)
    if args.overwrite is False:
        if os.path.exists(logs_path): raise Exception(RESULTS_EXIST_ERROR_MSG.format(logs_path))
        if os.path.exists(checkpoints_path): raise Exception(RESULTS_EXIST_ERROR_MSG.format(checkpoints_path))
        if os.path.exists(summary_path): raise Exception(RESULTS_EXIST_ERROR_MSG.format(summary_path))

    clean_dir(logs_path)
    clean_dir(checkpoints_path)

    # TODO: ensure write access to the directory

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

        # TODO: validate config
        assert not 'firelab' in config

        # Let's augment config with some helping stuff
        config['firelab'] = {}
        config['firelab']['project_path'] = os.getcwd()
        config['firelab']['logs_path'] = logs_path
        config['firelab']['checkpoints_path'] = checkpoints_path

        if 'random_seed' in config:
            torch.manual_seed(config['random_seed'])
            numpy.random.seed(config['random_seed'])

        # TODO: make config immutable

    # TODO: are there any better ways to reach src.trainers?
    sys.path.append(os.getcwd())
    trainers = importlib.import_module('src.trainers')
    trainer_cls = getattr(trainers, config.get('trainer'))
    trainer = trainer_cls(config)
    trainer.start()
