import argparse
from typing import List

from .manager import run


def main():
    parser = argparse.ArgumentParser(description='Run project commands')
    subparsers = parser.add_subparsers(dest='command')

    extend_with_start_parser(subparsers)
    extend_with_continue_parser(subparsers)
    extend_with_run_tb_parser(subparsers)

    args, _ = parser.parse_known_args()

    run(args.command, args)


def extend_with_arguments(parser: argparse.ArgumentParser, arguments_list: List[str]=[]):
    if 'experiment_dir' in arguments_list:
        parser.add_argument('--experiment_dir', type=str, help='A path to the experiment directory where to store the results.')

    if 'exp_name' in arguments_list:
        parser.add_argument('--exp_name', type=str, metavar='MY_EXPERIMENT', help='Experiment name')

    if 'tb_port' in arguments_list:
        parser.add_argument('--tb-port', type=int, help='Port for tensorboard')


def extend_with_start_parser(subparsers):
    "Augments parsers with a parser for `start` command"
    parser = subparsers.add_parser('start')
    parser.add_argument('config_path', type=str, metavar='config_path', help='Path to a config file')
    parser.add_argument('--stay-after-training', '-s', action='store_true') # TODO: rename.
    extend_with_arguments(parser, ['experiment_dir', 'exp_name', 'tb_port'])


def extend_with_run_tb_parser(subparsers):
    "Augments parsers with a parser for `tb` command"
    parser = subparsers.add_parser('tb')
    extend_with_arguments(parser, ['experiment_dir', 'exp_name', 'tb_port'])


def extend_with_continue_parser(subparsers):
    "Augments parsers with a parser for `continue` command"
    parser = subparsers.add_parser('continue')
    parser.add_argument('--iteration', type=int, metavar='iteration',
        help='Iteration from which we should continue training.')
    # TODO: looks like we should better keep it in some firelab experiment state
    parser.add_argument('--reset-iters-counter', action='store_true',
        help='Should we reset iters counter or recalculate it from dataloader length?')
    extend_with_arguments(parser, ['experiment_dir', 'exp_name', 'tb_port'])



def extend_with_pause_parser(subparsers):
    "Augments parsers with a parser for `pause` command"
    raise NotImplementedError
