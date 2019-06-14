import argparse

from .manager import run


def main():
    parser = argparse.ArgumentParser(description='Run project commands')
    subparsers = parser.add_subparsers(dest='command')

    extend_with_start_parser(subparsers)
    extend_with_continue_parser(subparsers)
    extend_with_run_tb_parser(subparsers)
    extend_with_clean_parser(subparsers)

    args = parser.parse_args()

    run(args.command, args)


def extend_with_start_parser(subparsers):
    "Augments parsers with a parser for `start` command"
    parser = subparsers.add_parser('start')
    parser.add_argument('config_path', type=str, metavar='config_path', help='Path to a config file')
    parser.add_argument('--stay-after-training', '-s', action='store_true') # TODO: rename.
    parser.add_argument('--tb-port', type=int, help='Port for tensorboard')


def extend_with_run_tb_parser(subparsers):
    "Augments parsers with a parser for `tb` command"
    parser = subparsers.add_parser('tb')
    parser.add_argument('exp_name', type=str, metavar='exp_name',
        help='Directory name in `experiments` directory. '
        'Must contain config file to run the experiment.')
    parser.add_argument('--tb-port', type=int, help='Port for tensorboard', required=True)


def extend_with_continue_parser(subparsers):
    "Augments parsers with a parser for `continue` command"
    parser = subparsers.add_parser('continue')
    parser.add_argument('exp_name', type=str, metavar='exp_name',
        help='Directory name in `experiments` directory. '
        'Must contain config file to run the experiment.')
    parser.add_argument('--iteration', type=int, metavar='iteration',
        help='Iteration from which we should continue training.')
    parser.add_argument('--tb-port', type=int, help='Port for tensorboard')
    # TODO: looks like we should better keep it in some firelab experiment state
    parser.add_argument('--reset-iters-counter', action='store_true',
        help='Should we reset iters counter or recalculate it from dataloader length?')


def extend_with_pause_parser(subparsers):
    "Augments parsers with a parser for `pause` command"
    raise NotImplementedError


def extend_with_clean_parser(subparsers):
    # TODO: clean by time
    parser = subparsers.add_parser('clean')
    parser.add_argument('prefix', type=str, metavar='prefix',
        help='Removes all experiments in experiments/ dir with specified prefix')
