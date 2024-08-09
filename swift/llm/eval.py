# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import datetime as dt
import multiprocessing
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import json
from evalscope.backend.opencompass import OpenCompassBackendManager
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
from evalscope.config import TaskConfig
from evalscope.constants import DEFAULT_ROOT_CACHE_DIR
from evalscope.models.custom import CustomModel
from evalscope.run import run_task
from evalscope.summarizer import Summarizer
from evalscope.utils import EvalBackend
from modelscope import GenerationConfig
from openai import APIConnectionError
from tqdm import tqdm

from swift.utils import append_to_jsonl, get_logger, get_main, seed_everything
from .infer import merge_lora, prepare_model_template
from .utils import DeployArguments, EvalArguments, XRequestConfig, inference, inference_client_async

logger = get_logger()


class EvalModel(CustomModel):

    def __init__(self, args: EvalArguments, model_name: str, **kwargs) -> None:
        if args.eval_url is None:
            if args.merge_lora:
                merge_lora(args, device_map=args.merge_device_map)
            if args.infer_backend == 'vllm':
                from .utils import prepare_vllm_engine_template
                self.llm_engine, self.template = prepare_vllm_engine_template(args)
            else:
                self.model, self.template = prepare_model_template(args)

        self.args = args
        super().__init__(config={'model_id': model_name}, **kwargs)
        self.model_name = model_name

    @staticmethod
    async def _call_openai(model_type: str, query: str, eval_url: str, *, is_chat_model: bool,
                           request_config: XRequestConfig, prog_bar: tqdm) -> Tuple[str, Optional[int]]:
        # idx: maintain the order
        resp = await inference_client_async(
            model_type, query, is_chat_request=is_chat_model, request_config=request_config, url=eval_url)
        if is_chat_model:
            response = resp.choices[0].message.content
        else:
            response = resp.choices[0].text
        prog_bar.update()
        return response

    async def call_openai_batched(self, prompts: List[str], request_config: XRequestConfig) -> List[str]:
        assert self.args.eval_is_chat_model is not None
        use_tqdm = True if len(prompts) >= 20 else False
        prog_bar = tqdm(total=len(prompts), dynamic_ncols=True, disable=not use_tqdm)
        tasks = []
        for prompt in prompts:
            tasks.append(
                self._call_openai(
                    self.args.model_type,
                    prompt,
                    self.args.eval_url,
                    is_chat_model=self.args.eval_is_chat_model,
                    request_config=request_config,
                    prog_bar=prog_bar))
        response_list: List[Optional[str]] = await asyncio.gather(*tasks)
        prog_bar.close()
        return response_list

    def predict(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        infer_cfg = kwargs['infer_cfg'].copy()
        infer_cfg.pop('limit', None)
        infer_cfg.pop('max_length', None)
        assert infer_cfg.get('max_new_tokens') is not None, f'infer_cfg: {infer_cfg}'
        do_sample = infer_cfg.pop('do_sample', None)

        if self.args.eval_url is not None:
            if do_sample is False:
                infer_cfg['temperature'] = 0
            max_new_tokens = infer_cfg.pop('max_new_tokens', None)
            if max_new_tokens is not None:
                infer_cfg['max_tokens'] = max_new_tokens

            request_config = XRequestConfig(**infer_cfg)
            response_list = asyncio.run(self.call_openai_batched(prompts, request_config))

        elif self.args.infer_backend == 'vllm':
            from .utils import inference_vllm, VllmGenerationConfig
            if do_sample is False:
                infer_cfg['temperature'] = 0
            generation_config = VllmGenerationConfig(**infer_cfg)

            request_list = [{'query': prompt} for prompt in prompts]
            use_tqdm = True if len(request_list) >= 20 else False
            resp_list = inference_vllm(
                self.llm_engine, self.template, request_list, generation_config=generation_config, use_tqdm=use_tqdm)
            response_list = [resp['response'] for resp in resp_list]
        else:
            if do_sample is False:
                # fix warning
                infer_cfg['temperature'] = 1.
                infer_cfg['top_p'] = 1.
                infer_cfg['top_k'] = 50
            if do_sample is not None:
                infer_cfg['do_sample'] = do_sample
            response_list = []
            generation_config = GenerationConfig(**infer_cfg)
            use_tqdm = True if len(prompts) >= 5 else False
            prog_bar = tqdm(total=len(prompts), dynamic_ncols=True, disable=not use_tqdm)
            for prompt in prompts:
                response, _ = inference(self.model, self.template, prompt, generation_config=generation_config)
                response_list.append(response)
                prog_bar.update()
            prog_bar.close()
        res_d = []
        for response in response_list:
            res_d.append({
                'choices': [{
                    'index': 0,
                    'message': {
                        'content': response,
                        'role': 'assistant'
                    }
                }],
                'created': int(time.time()),
                'model': self.model_name,
                'object': 'chat.completion',
            })
        return res_d


def run_custom_model(args: EvalArguments):
    from swift.llm import deploy_main
    port = args.port
    args = args.__dict__
    attrs = dir(DeployArguments)
    for key in list(args.keys()):
        if key not in attrs:
            args.pop(key)
    args['verbose'] = False
    deploy_args = DeployArguments(**args)
    deploy_args.port = port
    deploy_main(deploy_args)


class EvalDatasetContext:

    def __init__(self):
        self.cache_dir = self.prepare_evalscope_dataset()

    def __enter__(self):
        data_dir = os.path.join(self.cache_dir, 'data')
        local_dir = os.path.join(os.getcwd(), 'data')
        if os.path.exists(local_dir) and not os.path.islink(local_dir):
            raise AssertionError('Please promise your pwd dir does not contain a `data` dir.')
        if os.path.islink(local_dir):
            os.remove(os.path.join(local_dir))
        os.symlink(data_dir, local_dir)

    def __exit__(self, *args, **kwargs):
        pass

    @staticmethod
    def prepare_evalscope_dataset():
        from swift.llm.utils.media import MediaCache
        return MediaCache.download(
            'https://www.modelscope.cn/api/v1/datasets/swift/evalscope_resource/'
            'repo?Revision=master&FilePath=eval.zip', 'evalscope')


def get_model_type(port, timeout):
    cnt = 0
    while True:
        from openai import OpenAI
        client = OpenAI(
            api_key='EMPTY',
            base_url=f'http://localhost:{port}/v1',
        )
        try:
            return client.models.list().data
        except APIConnectionError as e:
            cnt += 1
            if cnt > timeout:
                logger.error('Cannot get model_type from the deploy service, please check the error to continue eval')
                raise e
            else:
                time.sleep(1)


def opencompass_runner(args: EvalArguments, dataset: List[str], model_type: str, is_chat: bool, url: str):
    eval_limit = args.eval_limit
    if eval_limit is not None and '[' not in eval_limit:
        eval_limit = int(eval_limit)
    limit_config = {'limit': eval_limit} if eval_limit else {}
    task_cfg = dict(
        eval_backend='OpenCompass',
        eval_config={
            'datasets':
            dataset,
            'reuse':
            'latest' if args.eval_use_cache else None,
            'batch_size':
            args.eval_batch_size,
            'work_dir':
            args.eval_output_dir,
            'models': [
                {
                    'path': model_type,
                    'openai_api_base': url,
                    'is_chat': is_chat,
                    'key': args.eval_token,
                    'temperature': args.temperature
                },
            ],
            **limit_config,
        },
    )
    with EvalDatasetContext():
        run_task(task_cfg=task_cfg)

    return Summarizer.get_report_from_cfg(task_cfg=task_cfg)


def vlmeval_runner(args: EvalArguments, dataset: List[str], model_type: str, is_chat: bool, url: str):
    eval_limit = args.eval_limit
    if eval_limit is not None and '[' not in eval_limit:
        eval_limit = int(eval_limit)
    limit_config = {'limit': eval_limit} if eval_limit else {}
    if args.eval_batch_size or args.eval_use_cache:
        logger.warn('VLMEval does not support `eval_batch_size` or `eval_use_cache`')
    task_cfg = dict(
        eval_backend='VLMEvalKit',
        eval_config={
            'data':
            dataset,
            'work_dir':
            args.eval_output_dir,
            'model': [
                {
                    'name': 'CustomAPIModel',
                    'api_base': url,
                    'key': args.eval_token,
                    'type': model_type,
                    'temperature': args.temperature
                },
            ],
            **limit_config,
        },
    )
    run_task(task_cfg=task_cfg)
    return Summarizer.get_report_from_cfg(task_cfg=task_cfg)


def eval_opencompass(args: EvalArguments) -> List[Dict[str, Any]]:
    from evalscope.run import run_task
    from swift.utils.torch_utils import _find_free_port
    logger.info(f'args: {args}')
    if args.eval_few_shot:
        logger.warn('OpenCompass does not support `eval_few_shot`')
    process = None
    if not args.eval_url:
        seed_everything(args.seed)
        port = _find_free_port()
        args.port = port
        mp = multiprocessing.get_context('spawn')
        process = mp.Process(target=run_custom_model, args=(args, ))
        process.start()

        # health check: try to get model_type until raises
        get_model_type(port, args.deploy_timeout)
        model_type = 'default-lora' if args.sft_type in ('lora',
                                                         'longlora') and not args.merge_lora else args.model_type
        from .deploy import is_generation_template
        if is_generation_template(args.template_type):
            url = f'http://127.0.0.1:{port}/v1/completions'
        else:
            url = f'http://127.0.0.1:{port}/v1/chat/completions'
        is_chat = not is_generation_template(args.template_type)
    else:
        url = args.eval_url
        url = url.rstrip('/')
        if args.eval_is_chat_model:
            url += '/chat/completions'
        else:
            url += '/completions'
        model_type = args.model_type
        is_chat = args.eval_is_chat_model

    nlp_datasets = set(OpenCompassBackendManager.list_datasets()) & set(args.eval_dataset)
    mm_datasets = set(VLMEvalKitBackendManager.list_supported_datasets()) & set(args.eval_dataset)

    for dataset, runner in zip([list(nlp_datasets), list(mm_datasets)], [opencompass_runner, vlmeval_runner]):
        if not dataset:
            continue

        final_report = runner(args, dataset, model_type, is_chat, url)
        logger.info(f'Final report:{final_report}\n')
    if process:
        process.kill()
    return final_report


def eval_llmuses(args: EvalArguments) -> List[Dict[str, Any]]:
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    model_name = args.model_type
    tm = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name += f'-{args.name or tm}'
    custom_names = []
    if args.custom_eval_config is not None:
        assert os.path.isfile(args.custom_eval_config)
        with open(args.custom_eval_config, 'r') as f:
            custom_eval = json.load(f)
            for _ds in custom_eval:
                custom_names.append(_ds['name'])
                TaskConfig.registry(_ds['name'], _ds['pattern'], _ds['dataset'], subset_list=_ds.get('subset_list'))
    eval_model = EvalModel(args, model_name)

    generation_config = {
        'do_sample': args.do_sample,
        'repetition_penalty': args.repetition_penalty,
        'max_length': args.max_length,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
    }

    task_configs = TaskConfig.load(custom_model=eval_model, tasks=args.eval_dataset + custom_names)
    for task_config in task_configs:
        task_config.generation_config = generation_config
        task_config.dataset_dir = DEFAULT_ROOT_CACHE_DIR
        task_config.use_cache = args.eval_use_cache
        if args.eval_limit is not None:
            task_config.limit = int(args.eval_limit)
        eval_few_shot = args.eval_few_shot
        if 'mmlu' in task_config.datasets:
            eval_few_shot = 0  # fix
        if eval_few_shot is not None:
            for dataset in task_config.datasets:
                if not task_config.dataset_args.get(dataset):
                    task_config.dataset_args[dataset] = {}
                task_config.dataset_args[dataset]['few_shot_num'] = eval_few_shot

    run_task(task_cfg=task_configs)
    final_report: List[dict] = Summarizer.get_report_from_cfg(task_cfg=task_configs)
    logger.info(f'Final report:{final_report}\n')

    result_dir = os.path.join(args.eval_output_dir, tm)
    if result_dir is None:
        result_dir = eval_model.llm_engine.model_dir if args.infer_backend == 'vllm' else eval_model.model.model_dir
    assert result_dir is not None
    os.makedirs(result_dir, exist_ok=True)
    jsonl_path = os.path.join(result_dir, 'eval_result.jsonl')
    result = {report['name']: report['score'] for report in final_report}
    logger.info(f'result: {result}')
    result_info = {
        'result': result,
        'model': args.model_type,
        'time': tm,
    }
    append_to_jsonl(jsonl_path, result_info)
    logger.info(f'save_result_path: {jsonl_path}')
    return final_report


def llm_eval(args: EvalArguments) -> List[Dict[str, Any]]:
    args.eval_output_dir = os.path.join(args.eval_output_dir, args.name or 'default')
    if args.custom_eval_config:
        args.eval_backend = EvalBackend.NATIVE.value
        if args.eval_dataset:
            logger.warn('--custom_eval_config cannot use together with --eval_dataset')
            args.eval_dataset = []
    if args.eval_backend == EvalBackend.OPEN_COMPASS.value:
        return eval_opencompass(args)
    else:
        return eval_llmuses(args)


eval_main = get_main(EvalArguments, llm_eval)
