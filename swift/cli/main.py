# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import os
import subprocess
import sys
from typing import Dict, List, Optional

ROUTE_MAPPING: Dict[str, str] = {
    'sft': 'swift.cli.sft',
    'infer': 'swift.cli.infer',
    'app-ui': 'swift.cli.app_ui',
    'merge-lora': 'swift.cli.merge_lora',
    'web-ui': 'swift.cli.web_ui',
    'deploy': 'swift.cli.deploy',
    'dpo': 'swift.cli.dpo',
    'export': 'swift.cli.export',
    'eval': 'swift.cli.eval'
}

ROUTE_MAPPING.update(
    {k.replace('-', '_'): v
     for k, v in ROUTE_MAPPING.items()})


def use_torchrun() -> bool:
    nproc_per_node = os.getenv('NPROC_PER_NODE')
    nnodes = os.getenv('NNODES')
    if nproc_per_node is None and nnodes is None:
        return False
    return True


def get_torchrun_args() -> Optional[List[str]]:
    if not use_torchrun():
        return
    torchrun_args = []
    for env_key in [
            'NPROC_PER_NODE', 'MASTER_PORT', 'NNODES', 'NODE_RANK',
            'MASTER_ADDR'
    ]:
        env_val = os.getenv(env_key)
        if env_val is None:
            continue
        torchrun_args += [f'--{env_key.lower()}', env_val]
    return torchrun_args


def cli_main() -> None:
    argv = sys.argv[1:]
    method_name = argv[0]
    argv = argv[1:]
    file_path = importlib.util.find_spec(ROUTE_MAPPING[method_name]).origin
    torchrun_args = get_torchrun_args()
    if torchrun_args is None or method_name not in ('sft', 'dpo'):
        args = ['python', file_path, *argv]
    else:
        args = ['torchrun', *torchrun_args, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    subprocess.run(args)


if __name__ == '__main__':
    cli_main()
