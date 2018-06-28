import argparse

from .manager import run


def main():
    parser = argparse.ArgumentParser(description='Run project commands')
    subparsers = parser.add_subparsers(dest='command')
    add_start_parser(subparsers)
    add_continue_parser(subparsers)
    args = parser.parse_args()

    run(args.command, args)


def add_start_parser(subparsers):
    """Augments parsers with a parser for `start` command"""
    start_parser = subparsers.add_parser('start')
    start_parser.add_argument('name', type=str, metavar='name',
        help='Directory name in `experiments` directory. '
        'Must contain config file to run the experiment.')
    start_parser.add_argument('--overwrite', '-o', action='store_true')


def add_continue_parser(subparsers):
    # TODO: this is a duplication of `start` cmd. How to fix it?
    """Augments parsers with a parser for `continue` command"""
    start_parser = subparsers.add_parser('continue')
    start_parser.add_argument('name', type=str, metavar='name',
        help='Directory name in `experiments` directory. '
        'Must contain config file to run the experiment.')


def add_pause_parser(subparsers):
    """Augments parsers with a parser for `pause` command"""
    raise NotImplementedError
