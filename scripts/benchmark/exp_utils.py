import os
import shutil
import subprocess
import time
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from queue import Queue
from typing import Any, Dict, List

import json
import torch

from swift.llm import ExportArguments
from swift.utils import get_logger
from swift.utils.torch_utils import _find_free_port

logger = get_logger()


@dataclass
class Experiment:

    name: str

    cmd: str

    requirements: Dict = field(default_factory=dict)

    eval_requirements: Dict = field(default_factory=dict)

    eval_dataset: List = field(default_factory=list)

    args: Dict = field(default_factory=dict)

    env: Dict = field(default_factory=dict)

    record: Dict = field(default_factory=dict)

    create_time: float = None

    runtime: Dict = field(default_factory=dict)

    input_args: Any = None

    do_eval = False

    def __init__(self,
                 name,
                 cmd,
                 requirements=None,
                 eval_requirements=None,
                 eval_dataset=None,
                 args=None,
                 input_args=None,
                 **kwargs):
        self.name = name
        self.cmd = cmd
        self.requirements = requirements or {}
        self.args = args or {}
        self.record = {}
        self.env = {}
        self.runtime = {}
        self.input_args = input_args
        self.eval_requirements = eval_requirements or {}
        self.eval_dataset = eval_dataset or []

    def load(self, _json):
        self.name = _json['name']
        self.cmd = _json['cmd']
        self.requirements = _json['requirements']
        self.args = _json['args']
        self.record = _json['record']
        self.env = _json['env']
        self.create_time = _json['create_time']

    @property
    def priority(self):
        return self.requirements.get('gpu', 0)

    def to_dict(self):
        _dict = asdict(self)
        _dict.pop('runtime')
        _dict.pop('input_args')
        return _dict


class ExpManager:

    RESULT_FILE = 'result.jsonl'

    def __init__(self):
        self.exps = []

    def run(self, exp: Experiment):
        if os.path.exists(
                os.path.join(exp.input_args.save_dir, exp.name + '.json')):
            with open(
                    os.path.join(exp.input_args.save_dir, exp.name + '.json'),
                    'r') as f:
                _json = json.load(f)
                if exp.eval_dataset and 'eval_result' not in _json['record']:
                    if not exp.do_eval:
                        logger.info(
                            f'Experiment {exp.name} need eval, load from file.'
                        )
                        exp.load(_json)
                        exp.do_eval = True
                else:
                    logger.warn(f'Experiment {exp.name} already done, skip')
                    return

        if exp.do_eval:
            runtime = self._build_eval_cmd(exp)
            exp.runtime = runtime
            envs = deepcopy(runtime.get('env', {}))
            envs.update(os.environ)
            logger.info(
                f'Running cmd: {runtime["running_cmd"]}, env: {runtime.get("env", {})}'
            )
            os.makedirs('exp', exist_ok=True)
            log_file = os.path.join('exp', f'{exp.name}.eval.log')
            exp.handler = subprocess.Popen(
                runtime['running_cmd'] + f' > {log_file} 2>&1',
                env=envs,
                shell=True)
            self.exps.append(exp)
            return

        if any([exp.name == e.name for e in self.exps]):
            raise ValueError(f'Why exp name duplicate? {exp.name}')
        elif exp.cmd == 'export' and any(
            [exp.cmd == 'export' for exp in self.exps]):  # noqa
            raise AssertionError('Cannot run parallel export task.')
        else:
            exp.create_time = time.time()
            runtime = self._build_cmd(exp)
            exp.runtime = runtime
            envs = deepcopy(runtime.get('env', {}))
            envs.update(os.environ)
            logger.info(
                f'Running cmd: {runtime["running_cmd"]}, env: {runtime.get("env", {})}'
            )
            os.makedirs('exp', exist_ok=True)
            log_file = os.path.join('exp', f'{exp.name}.{exp.cmd}.log')
            exp.handler = subprocess.Popen(
                runtime['running_cmd'] + f' > {log_file} 2>&1',
                env=envs,
                shell=True)
            self.exps.append(exp)

    def _build_eval_cmd(self, exp: Experiment):
        gpu = exp.eval_requirements.get('gpu', None)
        env = {}
        allocated = []
        if gpu:
            allocated = self._find_free_gpu(int(gpu))
            assert allocated, 'No free gpu for now!'
            allocated = [str(gpu) for gpu in allocated]
            env['CUDA_VISIBLE_DEVICES'] = ','.join(allocated)

        best_model_checkpoint = exp.record.get('best_model_checkpoint')
        eval_dataset = exp.eval_dataset
        cmd = f'swift eval --ckpt_dir {best_model_checkpoint} --name {exp.name} --eval_dataset {" ".join(eval_dataset)}'
        return {
            'running_cmd': cmd,
            'gpu': allocated,
            'env': env,
        }

    def _build_cmd(self, exp: Experiment):
        gpu = exp.requirements.get('gpu', None)
        env = {}
        allocated = []
        if gpu:
            allocated = self._find_free_gpu(int(gpu))
            assert allocated, 'No free gpu for now!'
            allocated = [str(gpu) for gpu in allocated]
            env['CUDA_VISIBLE_DEVICES'] = ','.join(allocated)
        if int(exp.requirements.get('ddp', 1)) > 1:
            env['NPROC_PER_NODE'] = exp.requirements.get('ddp')
            env['MASTER_PORT'] = str(_find_free_port())

        if exp.cmd == 'sft':
            from swift.llm import SftArguments
            args = exp.args
            sft_args = SftArguments(**args)
            args['output_dir'] = sft_args.output_dir
            args['logging_dir'] = sft_args.logging_dir
            args['add_output_dir_suffix'] = False
            os.makedirs(sft_args.output_dir, exist_ok=True)
            os.makedirs(sft_args.logging_dir, exist_ok=True)
            cmd = 'swift sft '
            for key, value in args.items():
                cmd += f' --{key} {value}'
        elif exp.cmd == 'dpo':
            from swift.llm import DPOArguments
            args = exp.args
            dpo_args = DPOArguments(**args)
            args['output_dir'] = dpo_args.output_dir
            args['logging_dir'] = dpo_args.logging_dir
            args['add_output_dir_suffix'] = False
            os.makedirs(dpo_args.output_dir, exist_ok=True)
            os.makedirs(dpo_args.logging_dir, exist_ok=True)
            cmd = 'swift dpo '
            for key, value in args.items():
                cmd += f' --{key} {value}'
        elif exp.cmd == 'export':
            args = exp.args
            cmd = 'swift export '
            for key, value in args.items():
                cmd += f' --{key} {value}'
        else:
            raise ValueError(f'Unsupported cmd type: {exp.cmd}')
        return {
            'running_cmd': cmd,
            'gpu': allocated,
            'env': env,
            'logging_dir': args.get('logging_dir'),
            'output_dir': args.get('output_dir', args.get('ckpt_dir'))
        }

    def _find_free_gpu(self, n):
        all_gpus = set()
        for exp in self.exps:
            all_gpus.update(exp.runtime.get('gpu', set()))
        all_gpus = {int(g) for g in all_gpus}
        free_gpu = set(range(torch.cuda.device_count())) - all_gpus
        if len(free_gpu) < n:
            return None
        return list(free_gpu)[:n]

    def prepare_experiments(self, args: Any):
        experiments = []
        for config_file in args.config:
            with open(config_file, 'r') as f:
                content = json.load(f)
                exps = content['experiment']
                for exp in exps:
                    main_cfg = deepcopy(content)
                    name = exp['name']
                    cmd = main_cfg['cmd']
                    run_args = main_cfg['args']
                    env = main_cfg.get('env', {})
                    requirements = main_cfg.get('requirements', {})
                    eval_requirements = main_cfg.get('eval_requirements', {})
                    eval_dataset = main_cfg.get('eval_dataset', {})
                    if 'args' in exp:
                        run_args.update(exp['args'])
                    if 'requirements' in exp:
                        requirements.update(exp['requirements'])
                    if 'env' in exp:
                        env.update(exp['env'])
                    experiments.append(
                        Experiment(
                            name=name,
                            cmd=cmd,
                            args=run_args,
                            env=env,
                            requirements=requirements,
                            eval_requirements=eval_requirements,
                            eval_dataset=eval_dataset,
                            input_args=args))
        return experiments

    @staticmethod
    def _get_metric(exp: Experiment):
        if exp.do_eval:
            if os.path.isfile(f'{exp.name}.eval.log'):
                with open(f'{exp.name}.eval.log', 'r') as f:
                    for line in f.readlines():
                        if 'Final report:' in line:
                            return json.loads(
                                line.split('Final report:')[1].replace(
                                    '\'', '"'))
        elif exp.cmd == 'export':
            exp_args = ExportArguments(**exp.args)
            if exp_args.quant_bits > 0:
                if exp_args.ckpt_dir is None:
                    path = f'{exp_args.model_type}-{exp_args.quant_method}-int{exp_args.quant_bits}'
                else:
                    ckpt_dir, ckpt_name = os.path.split(exp_args.ckpt_dir)
                    path = os.path.join(
                        ckpt_dir,
                        f'{ckpt_name}-{exp_args.quant_method}-int{exp_args.quant_bits}'
                    )
            else:
                ckpt_dir, ckpt_name = os.path.split(exp_args.ckpt_dir)
                path = os.path.join(ckpt_dir, f'{ckpt_name}-merged')
            if os.path.exists(path):
                shutil.rmtree(exp.name, ignore_errors=True)
                os.makedirs(exp.name, exist_ok=True)
                shutil.move(path, os.path.join(exp.name, path))
                return {
                    'best_model_checkpoint': os.path.join(exp.name, path),
                }
        else:
            logging_dir = exp.runtime.get('logging_dir')
            logging_file = os.path.join(logging_dir, '..', 'logging.jsonl')
            if os.path.isfile(logging_file):
                with open(logging_file, 'r') as f:
                    for line in f.readlines():
                        if 'model_info' in line:
                            return json.loads(line)
        return None

    @staticmethod
    def write_record(exp: Experiment):
        target_dir = exp.input_args.save_dir
        file = os.path.join(target_dir, exp.name + '.json')
        with open(file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(exp.to_dict()) + '\n')

    def _poll(self):
        while True:
            time.sleep(5)

            has_finished = False
            for exp in self.exps:
                rt = exp.handler.poll()
                if rt is None:
                    continue

                has_finished = True
                if rt == 0:
                    if not exp.do_eval:
                        all_metric = self._get_metric(exp)
                        if all_metric:
                            exp.record.update(all_metric)
                            if exp.eval_dataset:
                                exp.do_eval = True
                                self.exp_queue.appendleft(exp)
                            self.write_record(exp)
                        else:
                            logger.error(
                                f'Running {exp.name} task, but no result found'
                            )
                    else:
                        all_metric = self._get_metric(exp)
                        exp.record['eval_result'] = all_metric
                        if all_metric:
                            self.write_record(exp)
                        else:
                            logger.error(
                                f'Running {exp.name} eval task, but no eval result found'
                            )
                logger.info(
                    f'Running {exp.name} finished with return code: {rt}')

            if has_finished:
                self.exps = [
                    exp for exp in self.exps if exp.handler.poll() is None
                ]
                break

    def begin(self, args: Any):
        exps = self.prepare_experiments(args)
        logger.info(f'all exps: {exps}')
        exps.sort(key=lambda e: e.priority)
        self.exp_queue = deque()
        for exp in exps:
            self.exp_queue.append(exp)

        while len(self.exp_queue) or len(self.exps) > 0:
            while len(self.exp_queue):
                try:
                    logger.info(f'Running exp: {self.exp_queue[0].name}')
                    self.run(self.exp_queue[0])
                except Exception as e:
                    if not isinstance(e, AssertionError):
                        logger.error(
                            f'Adding exp {self.exp_queue[0].name} error because of:'
                        )
                        logger.error(e)
                        self.exp_queue.popleft()
                    else:
                        logger.info(
                            f'Adding exp {self.exp_queue[0].name} error because of no free gpu.'
                        )
                    break
                else:
                    self.exp_queue.popleft()
            self._poll()
        logger.info(
            f'Run task finished because of exp queue: {self.exp_queue} and exps: {self.exps}'
        )


def find_all_config(dir_or_file: str):
    if os.path.isfile(dir_or_file):
        return [dir_or_file]
    else:
        configs = []
        for dirpath, dirnames, filenames in os.walk(dir_or_file):
            for name in filenames:
                if name.endswith('.json'):
                    configs.append(os.path.join(dirpath, name))
        return configs
