import argparse

from .manager import start_experiment

def main():
    parser = argparse.ArgumentParser(description='Run project commands')
    subparsers = parser.add_subparsers(dest='command')

    add_start_parser(subparsers)

    args = parser.parse_args()

    if args.command == 'start':
        # Starting experiment
        start_experiment(args)

    elif args.command == 'ls':
        # List current experiments
        raise NotImplementedError
    else:
        raise NotImplementedError


def add_start_parser(subparsers):
    """Augments parsers with a parser for `start` command"""
    start_parser = subparsers.add_parser('start')
    start_parser.add_argument('name', type=str, metavar='name',
        help='Directory name in `experiments` directory. '
        'Must contain config file to run the experiment.')
    start_parser.add_argument('--overwrite', '-o', action='store_true')


def add_pause_parser(subparsers):
    """Augments parsers with a parser for `pause` command"""
    raise NotImplementedError
