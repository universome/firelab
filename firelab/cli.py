import argparse

from .manager import run


def main():
    parser = argparse.ArgumentParser(description='Run project commands')
    subparsers = parser.add_subparsers(dest='command')
    extend_with_start_parser(subparsers)
    extend_with_continue_parser(subparsers)
    extend_with_touch_parser(subparsers)
    args = parser.parse_args()

    run(args.command, args)


def extend_with_start_parser(subparsers):
    "Augments parsers with a parser for `start` command"
    parser = subparsers.add_parser('start')
    parser.add_argument('name', type=str, metavar='name',
        help='Directory name in `experiments` directory. '
        'Must contain config file to run the experiment.')
    parser.add_argument('--overwrite', '-o', action='store_true')
    parser.add_argument('--tb-port', type=int, help='Port for tensorboard')


def extend_with_continue_parser(subparsers):
    # TODO: this is a duplication of `start` cmd. How to fix it?
    "Augments parsers with a parser for `continue` command"
    parser = subparsers.add_parser('continue')
    parser.add_argument('name', type=str, metavar='name',
        help='Directory name in `experiments` directory. '
        'Must contain config file to run the experiment.')
    parser.add_argument('--tb-port', type=int, help='Port for tensorboard')


def extend_with_touch_parser(subparsers):
    "Augments parsers with a parser for `touch` command"
    parser = subparsers.add_parser('touch')
    parser.add_argument('name', type=str, metavar='name', help='Name of the experiment')


def extend_with_pause_parser(subparsers):
    "Augments parsers with a parser for `pause` command"
    raise NotImplementedError
