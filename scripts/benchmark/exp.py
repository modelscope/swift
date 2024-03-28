# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import os.path

from exp_utils import ExpManager, find_all_config

from swift.utils import *

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple args for swift experiments.')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        required=True,
        help='The experiment config file',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./experiment',
        required=False,
        help='The experiment output folder',
    )

    args = parser.parse_args()
    return args


def llm_exp():
    args = parse_args()
    config: str = args.config
    config = config.split(',')
    os.makedirs(args.save_dir, exist_ok=True)
    all_configs = []
    if not isinstance(config, list):
        config = [config]
    for dir_or_file in config:
        all_configs.extend(find_all_config(dir_or_file))
    args.config = all_configs
    exp_manager = ExpManager()
    exp_manager.begin(args)


if __name__ == '__main__':
    llm_exp()
