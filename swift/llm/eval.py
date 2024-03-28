# Copyright (c) Alibaba, Inc. and its affiliates.
import dataclasses
import time
from typing import Any, Dict, List

import json
from llmuses.models.custom import CustomModel
from modelscope import GenerationConfig

from swift.utils import get_logger, get_main
from swift.utils.utils import split_str_parts_by
from . import (EvalArguments, inference, inference_vllm, merge_lora,
               prepare_model_template)
from .utils.model import dtype_mapping

logger = get_logger()


@dataclasses.dataclass
class ModelOutput:
    name: str = None

    cmd: str = None

    requirements: Dict[str, str] = dataclasses.field(default_factory=dict)

    args: Dict[str, Any] = dataclasses.field(default_factory=dict)

    memory: str = None

    train_time: float = None

    train_samples: int = None

    train_samples_per_second: float = None

    last_model_checkpoint: str = None

    best_model_checkpoint: str = None

    best_metric: Any = None

    global_step: int = None

    num_total_parameters: float = None

    num_trainable_parameters: float = None

    num_buffers: float = None

    trainable_parameters_percentage: float = None

    train_dataset_info: str = None

    val_dataset_info: str = None

    train_create_time: float = None


def parse_output(file):
    with open(file, 'r') as f:
        content = json.load(f)

    name = content['name']
    cmd = content['cmd']
    requirements = content['requirements']
    args = content['args']
    create_time = float(content['create_time'])
    content = content['record']
    if cmd == 'export':
        best_model_checkpoint = content['best_model_checkpoint']
    else:
        memory = content['memory']
        total_memory = 0.0
        for values in memory.values():
            total_memory += float(values.split('GiB')[0])
        memory = f'{total_memory}GiB'
        train_time = content['train_time']['train_runtime']
        train_samples = content['train_time']['n_train_samples']
        train_samples_per_second = content['train_time'][
            'train_samples_per_second']
        last_model_checkpoint = content['last_model_checkpoint']
        best_model_checkpoint = content['best_model_checkpoint']
        best_metric = content['best_metric']
        global_step = content['global_step']
        train_dataset_info = content['dataset_info']['train_dataset']
        val_dataset_info = content['dataset_info']['val_dataset']
        # model_info like: SwiftModel: 6758.4041M Params (19.9885M Trainable [0.2958%]), 16.7793M Buffers.
        str_dict = split_str_parts_by(content['model_info'], [
            'SwiftModel:', 'CausalLM:', 'Seq2SeqLM:', 'M Params (',
            'M Trainable [', ']), ', 'M Buffers.'
        ])
        str_dict = {c['key']: c['content'] for c in str_dict}
        if 'SwiftModel:' in str_dict:
            num_total_parameters = float(str_dict['SwiftModel:'])
        elif 'CausalLM:' in str_dict:
            num_total_parameters = float(str_dict['CausalLM:'])
        else:
            num_total_parameters = float(str_dict['Seq2SeqLM:'])
        num_trainable_parameters = float(str_dict['M Params ('])
        num_buffers = float(str_dict[']), '])
        trainable_parameters_percentage = str_dict['M Trainable [']

    return ModelOutput(
        name=name,
        cmd=cmd,
        requirements=requirements,
        args=args,
        memory=memory,
        train_time=train_time,
        train_samples=train_samples,
        train_samples_per_second=train_samples_per_second,
        last_model_checkpoint=last_model_checkpoint,
        best_model_checkpoint=best_model_checkpoint,
        best_metric=best_metric,
        global_step=global_step,
        train_dataset_info=train_dataset_info,
        val_dataset_info=val_dataset_info,
        train_create_time=create_time,
        num_total_parameters=num_total_parameters,
        num_trainable_parameters=num_trainable_parameters,
        num_buffers=num_buffers,
        trainable_parameters_percentage=trainable_parameters_percentage,
    )


class EvalModel(CustomModel):

    def __init__(self, args: EvalArguments, model_name, config={}, **kwargs):
        if args.merge_lora:
            merge_lora(args, device_map=args.merge_device_map)
        if args.infer_backend == 'vllm':
            from .utils import prepare_vllm_engine_template
            self.llm_engine, self.template = prepare_vllm_engine_template(args)
            self.max_model_len = self.model.config.max_position_embeddings
        else:
            self.model, self.template = prepare_model_template(args)
            self.max_model_len = self.model.config.max_position_embeddings
            if args.overwrite_generation_config:
                assert args.ckpt_dir is not None, 'args.ckpt_dir is not specified.'
                self.model.generation_config.save_pretrained(args.ckpt_dir)

        self.args = args
        super(EvalModel, self).__init__(
            config={
                'model_id': model_name,
                **config
            }, **kwargs)
        self.model_name = model_name

    def predict(self, prompt: str, **kwargs):
        if self.args.infer_backend == 'vllm':
            request_list = [{
                'query': prompt,
                'history': kwargs.get('history'),
                'system': kwargs.get('system')
            }]
            if 'temperature' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.temperature = kwargs[
                    'infer_cfg']['temperature']
            if 'max_new_tokens' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.max_new_tokens = kwargs[
                    'infer_cfg']['max_new_tokens']
            if 'top_k' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.top_k = kwargs['infer_cfg'][
                    'top_k']
            if 'top_p' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.top_p = kwargs['infer_cfg'][
                    'top_p']
            if 'repetition_penalty' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.repetition_penalty = kwargs[
                    'infer_cfg']['repetition_penalty']
            resp_list = inference_vllm(self.llm_engine, self.template,
                                       request_list)
            response = resp_list[0]['response']
            new_history = resp_list[0]['history']
        else:
            ts = time.time()
            response, new_history = inference(
                self.model,
                self.template,
                prompt,
                generation_config=GenerationConfig(**kwargs['infer_cfg']))
            print(time.time() - ts)

        res_d: dict = {
            'choices': [{
                'index': 0,
                'message': {
                    'content': response,
                    'role': 'assistant'
                }
            }],
            'created':
            int(time.time()),
            'model':
            self.model_name,
            'object':
            'chat.completion',
        }

        return res_d


def run_eval_single_model(args: EvalArguments, model_name, record=None):
    from llmuses.run import run_task
    from llmuses.config import TaskConfig
    from llmuses.summarizer import Summarizer
    eval_model = EvalModel(args, model_name, config=record or {})
    task_config: TaskConfig = TaskConfig()
    task_config = task_config.load(
        custom_model=eval_model, tasks=args.eval_dataset)
    if args.eval_limit:
        task_config.limit = args.eval_limit
    logger.warn('Eval does not support temperature/top_p/do_sample argument')
    logger.info(f'Eval task config: {task_config}')
    run_task(task_cfg=task_config)
    final_report: List[dict] = Summarizer.get_report_from_cfg(
        task_cfg=task_config)
    print(f'Final report:{final_report}\n', flush=True)
    return final_report


def llm_eval(args: EvalArguments) -> None:
    model_name = args.model_type
    if args.name:
        model_name += f'-{args.name}'
    run_eval_single_model(args, model_name)


eval_main = get_main(EvalArguments, llm_eval)
