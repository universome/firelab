import os
import shutil

import yaml

from ..config import Config


# TODO: move error msgs into separate file?
PATH_NOT_EXISTS_ERROR_MSG = ("`{}` directory or file does not exist")


def clean_dir(dirpath, create=False):
    if not os.path.exists(dirpath):
        if create: os.makedirs(dirpath, exist_ok=True)
        return

    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)

        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


def clean_file(filepath, create=False):
    if not os.path.exists(filepath):
        touch_file(filepath)
    else:
        open(filepath, 'w').close()


def touch_file(file_path):
    open(file_path, 'a').close()


def check_that_path_exists(path):
    if not os.path.exists(path):
        raise Exception(PATH_NOT_EXISTS_ERROR_MSG.format(path))


def check_that_path_does_not_exist(path):
    if os.path.exists(path):
        raise Exception(f"Path {path} already exists")


def infer_new_experiment_version(experiment_parent_dir: str, prefix: str) -> int:
    experiments = os.listdir(experiment_parent_dir)
    experiments = [exp for exp in experiments if exp.startswith(prefix)]
    versions = [exp[len(prefix) + 1:] for exp in experiments]
    versions= [v for v in versions if v.isdigit()]
    versions = [int(v) for v in versions]

    if len(versions) > 0:
        return max(versions) + 1
    else:
        return 1
