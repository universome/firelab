import os
import sys
import importlib.util

import yaml


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
    summary_path = os.path.join(experiments_dir, exp_name, "summary.md")

    if not os.path.isfile(config_path): raise FileNotFoundError(config_path)
    if args.overwrite is False:
        if os.path.exists(logs_path): raise Exception(RESULTS_EXIST_ERROR_MSG.format(logs_path))
        if os.path.exists(summary_path): raise Exception(RESULTS_EXIST_ERROR_MSG.format(summary_path))

    if not os.path.exists(logs_path): os.mkdir(logs_path)

    # TODO: ensure write access to the directory

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

        # TODO: validate config
        assert not 'firelab' in config

        # Let's augment config with some helping stuff
        config['firelab'] = {}
        config['firelab']['project_path'] = os.getcwd()

        # TODO: make config immutable

    # runner_path = path.join("src/runners/", config.get("runner") + ".py")
    # TODO: can be arbitrary? Can we have name collisions?
    # runner_module_name = "module.runner." + config.get("runner")
    # runner_module_spec = importlib.util.spec_from_file_location(runner_module_name, runner_path)
    # runner = importlib.util.module_from_spec(runner_module_spec)
    # runner_module_spec.loader.exec_module(runner)

    # TODO: are there any better ways to reach src.runners?
    sys.path.append(os.getcwd())
    runners = importlib.import_module('src.runners')
    runner_cls = getattr(runners, config.get('runner'))
    runner = runner_cls(config)
    runner.start()
