# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import json
import numpy as np
import torch
import torch.distributed as dist
import transformers
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from packaging import version
from torch import dtype as Dtype
from transformers.utils import (is_torch_bf16_gpu_available,
                                is_torch_cuda_available,
                                is_torch_npu_available)
from transformers.utils.versions import require_version

from swift.hub import HubApi, ModelScopeConfig
from swift.trainers import Seq2SeqTrainingArguments
from swift.tuners import Swift
from swift.utils import (add_version_to_work_dir, get_dist_setting, get_logger,
                         get_pai_tensorboard_dir, is_dist, is_mp,
                         is_pai_training_job)
from .dataset import (DATASET_MAPPING, get_custom_dataset, get_dataset,
                      register_dataset)
from .model import (MODEL_MAPPING, dtype_mapping, get_additional_saved_files,
                    get_default_lora_target_modules, get_default_template_type)
from .template import TEMPLATE_MAPPING, TemplateType
from .utils import is_vllm_available

logger = get_logger()


def is_adapter(sft_type: str) -> bool:
    return sft_type in {
        'lora', 'longlora', 'adalora', 'ia3', 'llamapro', 'adapter'
    }


@dataclass
class SftArguments:
    # You can specify the model by either using the model_type or model_id_or_path.
    model_type: Optional[str] = field(
        default=None,
        metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    model_id_or_path: Optional[str] = None
    model_revision: Optional[str] = None

    sft_type: Literal['lora', 'full', 'longlora', 'adalora', 'ia3', 'llamapro',
                      'adapter'] = 'lora'
    freeze_parameters: float = 0.  # 0 ~ 1
    additional_trainable_parameters: List[str] = field(default_factory=list)
    tuner_backend: Literal['swift', 'peft'] = 'peft'
    template_type: str = field(
        default='AUTO',
        metadata={
            'help':
            f"template_type choices: {list(TEMPLATE_MAPPING.keys()) + ['AUTO']}"
        })
    output_dir: str = 'output'
    add_output_dir_suffix: Optional[bool] = None
    ddp_backend: Literal['nccl', 'gloo', 'mpi', 'ccl'] = 'nccl'
    ddp_find_unused_parameters: Optional[bool] = None
    ddp_broadcast_buffers: Optional[bool] = None

    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    dtype: Literal['bf16', 'fp16', 'fp32', 'AUTO'] = 'AUTO'

    dataset: List[str] = field(
        default_factory=list,
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: int = 42
    dataset_test_ratio: float = 0.01
    train_dataset_sample: int = -1  # -1: all dataset
    train_dataset_mix_ratio: float = 0.
    train_dataset_mix_ds: List[str] = field(
        default_factory=lambda: ['ms-bench'])
    val_dataset_sample: Optional[int] = None  # -1: all dataset
    use_loss_scale: bool = False
    system: Optional[str] = None
    max_length: int = 2048  # -1: no limit
    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete'
    check_dataset_strategy: Literal['none', 'discard', 'error',
                                    'warning'] = 'none'
    custom_train_dataset_path: List[str] = field(default_factory=list)
    custom_val_dataset_path: List[str] = field(default_factory=list)
    self_cognition_sample: int = 0
    # Chinese name and English name
    model_name: List[str] = field(
        default_factory=lambda: [None, None],
        metadata={'help': "e.g. ['小黄', 'Xiao Huang']"})
    model_author: List[str] = field(
        default_factory=lambda: [None, None],
        metadata={'help': "e.g. ['魔搭', 'ModelScope']"})
    # If you want to use qlora, set the quantization_bit to 8 or 4.
    # And you need to install bitsandbytes: `pip install bitsandbytes -U`
    # note: bf16 and quantization have requirements for gpu architecture
    quantization_bit: Literal[0, 4, 8] = 0
    bnb_4bit_comp_dtype: Literal['fp16', 'bf16', 'fp32', 'AUTO'] = 'AUTO'
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    # lora
    lora_target_modules: List[str] = field(default_factory=lambda: ['DEFAULT'])
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.05
    lora_bias_trainable: Literal['none', 'all'] = 'none'
    # e.g. ['wte', 'ln_1', 'ln_2', 'ln_f', 'lm_head']
    lora_modules_to_save: List[str] = field(default_factory=list)
    lora_dtype: Literal['fp16', 'bf16', 'fp32', 'AUTO'] = 'fp32'
    lora_lr_ratio: float = None
    use_rslora: bool = False
    use_dora: bool = False

    # adapter
    adapter_act: str = 'gelu'
    adapter_length: int = 128

    # galore
    use_galore: bool = False
    galore_rank: int = 128
    galore_target_modules: Optional[List[str]] = None
    galore_update_proj_gap: int = 50
    galore_scale: float = 1.0
    galore_proj_type: str = 'std'
    galore_optim_per_parameter: bool = False
    galore_with_embedding: bool = False

    # adalora
    adalora_target_r: int = 8
    adalora_init_r: int = 12
    adalora_tinit: int = 0
    adalora_tfinal: int = 0
    adalora_deltaT: int = 1
    adalora_beta1: float = 0.85
    adalora_beta2: float = 0.85
    adalora_orth_reg_weight: float = 0.5
    # ia3
    ia3_target_modules: List[str] = field(default_factory=lambda: ['DEFAULT'])
    ia3_feedforward_modules: List[str] = field(default_factory=list)
    ia3_modules_to_save: List[str] = field(default_factory=list)
    # llamapro
    llamapro_num_new_blocks: int = 4
    llamapro_num_groups: Optional[int] = None

    # neftune
    neftune_noise_alpha: Optional[float] = None  # e.g. 5, 10, 15
    neftune_backend: Literal['swift', 'transformers'] = None

    gradient_checkpointing: Optional[bool] = None
    # e.g. 'default-zero3', 'default-zero2', 'ds_config/zero2.json'
    deepspeed: Optional[str] = None
    batch_size: int = 1
    eval_batch_size: Optional[int] = None
    num_train_epochs: int = 1
    # if max_steps >= 0, override num_train_epochs
    max_steps: int = -1
    optim: str = 'adamw_torch'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    learning_rate: Optional[float] = None
    weight_decay: float = 0.1
    gradient_accumulation_steps: Optional[int] = None
    max_grad_norm: float = 0.5
    predict_with_generate: bool = False
    lr_scheduler_type: str = 'linear'
    warmup_ratio: float = 0.05

    eval_steps: int = 50
    save_steps: Optional[int] = None
    save_only_model: Optional[bool] = None
    save_total_limit: int = 2  # save last and best. -1: all checkpoints
    logging_steps: int = 5
    dataloader_num_workers: int = 1
    dataloader_pin_memory: bool = True

    # push to ms hub
    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'SDK token can be found in https://modelscope.cn/my/myaccesstoken'
        })
    hub_private_repo: bool = False
    push_hub_strategy: Literal['end', 'push_best', 'push_last', 'checkpoint',
                               'all_checkpoints'] = 'push_best'

    # other
    test_oom_error: bool = field(
        default=False,
        metadata={
            'help':
            'If set to True, the train_dataset will be sorted in descending order based on max_length, '
            'enabling faster detection of OOM (Out of Memory) errors.'
        })
    disable_tqdm: bool = False
    lazy_tokenize: Optional[bool] = None
    preprocess_num_proc: int = 1
    use_flash_attn: Optional[bool] = None
    ignore_args_error: bool = False  # True: notebook compatibility
    check_model_is_latest: bool = True

    logging_dir: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    acc_strategy: Literal['token', 'sentence'] = 'token'
    save_on_each_node: bool = True
    evaluation_strategy: Literal['steps', 'no'] = 'steps'
    save_strategy: Literal['steps', 'no'] = 'steps'
    save_safetensors: bool = True
    gpu_memory_fraction: Optional[float] = None

    # generation config
    max_new_tokens: int = 2048
    do_sample: bool = True
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.7
    repetition_penalty: float = 1.
    num_beams: int = 1
    # compatibility hf
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    # compatibility. (Deprecated)
    only_save_model: Optional[bool] = None
    neftune_alpha: Optional[float] = None
    deepspeed_config_path: Optional[str] = None
    model_cache_dir: Optional[str] = None

    def _prepare_target_modules(self, target_modules) -> List[str]:
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        if len(target_modules) == 0:
            return target_modules
        elif len(target_modules) == 1:
            if ',' in target_modules[0]:
                target_modules = target_modules[0].split(',')
        if 'AUTO' in target_modules:
            target_modules.remove('AUTO')
            target_modules.append('DEFAULT')
        if 'DEFAULT' in target_modules:
            target_modules.remove('DEFAULT')
            target_modules += get_default_lora_target_modules(self.model_type)
        if 'EMBEDDING' in target_modules:
            target_modules.remove('EMBEDDING')
            self.lora_use_embedding = True
        if 'ALL' in target_modules:
            target_modules.remove('ALL')
            self.lora_use_all = True
        return target_modules

    def _prepare_modules_to_save(self, modules_to_save) -> List[str]:
        if isinstance(modules_to_save, str):
            modules_to_save = [modules_to_save]
        if len(modules_to_save) == 0:
            return modules_to_save
        if 'EMBEDDING' in modules_to_save:
            modules_to_save.remove('EMBEDDING')
            self.lora_m2s_use_embedding = True
        if 'LN' in modules_to_save:
            modules_to_save.remove('LN')
            self.lora_m2s_use_ln = True
        return modules_to_save

    def __post_init__(self) -> None:
        handle_compatibility(self)
        if is_pai_training_job():
            self._handle_pai_compat()
        ds_config_folder = os.path.join(__file__, '..', '..', 'ds_config')
        if self.deepspeed == 'default-zero2':
            self.deepspeed = os.path.abspath(
                os.path.join(ds_config_folder, 'zero2.json'))
        elif self.deepspeed == 'default-zero3':
            self.deepspeed = os.path.abspath(
                os.path.join(ds_config_folder, 'zero3.json'))
        handle_path(self)
        set_model_type(self)
        if isinstance(self.dataset, str):
            self.dataset = [self.dataset]
        if isinstance(self.train_dataset_mix_ds, str):
            self.train_dataset_mix_ds = [self.train_dataset_mix_ds]
        register_custom_dataset(self)
        check_flash_attn(self)
        handle_generation_config(self)

        self.lora_use_embedding = False
        self.lora_use_all = False
        self.lora_m2s_use_embedding = False
        self.lora_m2s_use_ln = False
        if self.sft_type == 'ia3':
            self.ia3_feedforward_modules = self._prepare_target_modules(
                self.ia3_feedforward_modules)
            self.ia3_target_modules = self._prepare_target_modules(
                self.ia3_target_modules)
            self.ia3_modules_to_save = self._prepare_modules_to_save(
                self.ia3_modules_to_save)
        else:
            self.lora_target_modules = self._prepare_target_modules(
                self.lora_target_modules)
            self.lora_modules_to_save = self._prepare_modules_to_save(
                self.lora_modules_to_save)
        if self.sft_type in {'adalora', 'ia3'} and self.lora_use_embedding:
            raise ValueError(
                '`adalora` and `ia3` do not support setting embedding as target_modules.'
            )

        if self.self_cognition_sample > 0:
            if self.model_name is None or self.model_author is None:
                raise ValueError(
                    'Please enter self.model_name self.model_author. '
                    'For example: `--model_name 小黄 "Xiao Huang" --model_author 魔搭 ModelScope`. '
                    'Representing the model name and model author in Chinese and English.'
                )
            for k in ['model_name', 'model_author']:
                v = getattr(self, k)
                if len(v) == 1:
                    v = v[0]
                if isinstance(v, str):
                    setattr(self, k, [v, v])
            if self.sft_type == 'lora' and not self.lora_use_all:
                logger.warning(
                    'Due to knowledge editing involved, it is recommended to add LoRA on MLP. '
                    'For example: `--lora_target_modules ALL`. '
                    'If you have already added LoRA on MLP, please ignore this warning.'
                )

        self.torch_dtype, self.fp16, self.bf16 = select_dtype(self)
        world_size = 1
        if is_dist():
            rank, local_rank, world_size, _ = get_dist_setting()
            torch.cuda.set_device(local_rank)
            self.seed += rank  # Avoid the same dropout
            if self.ddp_backend == 'gloo' and self.quantization_bit != 0:
                raise ValueError('not supported, please use `nccl`')

        if is_adapter(self.sft_type):
            assert self.freeze_parameters == 0., (
                'lora does not support `freeze_parameters`, please set `--sft_type full`'
            )
            assert len(self.additional_trainable_parameters) == 0, (
                'lora does not support `additional_trainable_parameters`, please set `--sft_type full`'
            )
            if 'int4' in self.model_type or 'int8' in self.model_type:
                assert self.quantization_bit == 0, 'int4 and int8 models do not need to be quantized again.'
            if self.learning_rate is None:
                self.learning_rate = 1e-4
            if self.save_only_model is None:
                if self.deepspeed is None:
                    self.save_only_model = False
                else:
                    self.save_only_model = True
        elif self.sft_type == 'full':
            assert 0 <= self.freeze_parameters <= 1
            assert self.quantization_bit == 0, 'Full parameter fine-tuning does not support quantization.'
            assert self.dtype != 'fp16', (
                "Fine-tuning with dtype=='fp16' can lead to NaN issues. "
                'Please use fp32+AMP or bf16 to perform full parameter fine-tuning.'
            )
            if isinstance(self.additional_trainable_parameters, str):
                self.additional_trainable_parameters = [
                    self.additional_trainable_parameters
                ]
            if self.learning_rate is None:
                self.learning_rate = 1e-5
            if self.save_only_model is None:
                self.save_only_model = True
        else:
            raise ValueError(f'sft_type: {self.sft_type}')

        if self.template_type == 'AUTO':
            self.template_type = get_default_template_type(self.model_type)
            logger.info(f'Setting template_type: {self.template_type}')
        if len(self.dataset) == 0 and (len(self.custom_train_dataset_path) == 0
                                       and len(
                                           self.custom_val_dataset_path) == 0
                                       and self.self_cognition_sample == 0):
            raise ValueError(
                f'self.dataset: {self.dataset}, Please input the training dataset.'
            )

        if self.save_steps is None:
            self.save_steps = self.eval_steps
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self)

        if self.neftune_backend is None:
            self.neftune_backend = 'swift' if version.parse(transformers.__version__) < version.parse('4.35') \
                else 'transformers'

        prepare_push_ms_hub(self)
        self.train_sampler_random = not self.test_oom_error
        if self.eval_batch_size is None:
            if self.predict_with_generate:
                self.eval_batch_size = 1
            else:
                self.eval_batch_size = self.batch_size
        if self.save_total_limit == -1:
            self.save_total_limit = None
        if self.max_length == -1:
            self.max_length = None

        if self.deepspeed is not None:
            if is_mp():
                raise ValueError('DeepSpeed is not compatible with MP. '
                                 f'n_gpu: {torch.cuda.device_count()}, '
                                 f'local_world_size: {get_dist_setting()[3]}.')
            require_version('deepspeed')
            if self.deepspeed.endswith('.json') or os.path.isfile(
                    self.deepspeed):
                with open(self.deepspeed, 'r', encoding='utf-8') as f:
                    self.deepspeed = json.load(f)
            logger.info(f'Using deepspeed: {self.deepspeed}')

        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = math.ceil(16 / self.batch_size
                                                         / world_size)
        template_info = TEMPLATE_MAPPING[self.template_type]
        if self.lazy_tokenize is None:
            self.lazy_tokenize = template_info.get('lazy_tokenize', False)
            logger.info(f'Setting args.lazy_tokenize: {self.lazy_tokenize}')
        if 'dataloader_num_workers' in template_info:
            self.dataloader_num_workers = template_info[
                'dataloader_num_workers']
            logger.info(
                f'Setting args.dataloader_num_workers: {self.dataloader_num_workers}'
            )
        if 'dataloader_pin_memory' in template_info:
            self.dataloader_pin_memory = template_info['dataloader_pin_memory']
            logger.info(
                f'Setting args.dataloader_pin_memory: {self.dataloader_pin_memory}'
            )
        if 'qwen-audio' in self.model_type:
            assert self.preprocess_num_proc == 1 or self.lazy_tokenize, 'not support'
        model_info = MODEL_MAPPING[self.model_type]
        support_gradient_checkpointing = model_info.get(
            'support_gradient_checkpointing', True)
        if self.gradient_checkpointing is None:
            self.gradient_checkpointing = support_gradient_checkpointing
        elif not support_gradient_checkpointing and self.gradient_checkpointing:
            logger.warning(
                f'{self.model_type} not support gradient_checkpointing.')

        self._init_training_args()

        if self.add_output_dir_suffix is None:
            self.add_output_dir_suffix = True
        if self.add_output_dir_suffix:
            self.output_dir = os.path.join(self.output_dir, self.model_type)
            self.output_dir = add_version_to_work_dir(self.output_dir)
            logger.info(f'output_dir: {self.output_dir}')
            self.training_args.output_dir = self.output_dir
            self.training_args.run_name = self.output_dir

        if self.logging_dir is None:
            self.logging_dir = f'{self.output_dir}/runs'
            self.training_args.logging_dir = self.logging_dir

    def _init_training_args(self) -> None:
        additional_saved_files = []
        if self.sft_type == 'full':
            additional_saved_files = get_additional_saved_files(
                self.model_type)

        kwargs = {}
        if self.neftune_backend != 'swift':
            kwargs['neftune_noise_alpha'] = self.neftune_noise_alpha

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy=self.evaluation_strategy,
            logging_dir=self.logging_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            remove_unused_columns=False,
            bf16=self.bf16,
            fp16=self.fp16,
            eval_steps=self.eval_steps,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
            metric_for_best_model='rouge-l'
            if self.predict_with_generate else 'loss',
            greater_is_better=self.predict_with_generate,
            sortish_sampler=True,
            optim=self.optim,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            hub_model_id=self.hub_model_id,
            hub_private_repo=self.hub_private_repo,
            push_hub_strategy=self.push_hub_strategy,
            hub_token=self.hub_token,
            push_to_hub=self.push_to_hub,
            resume_from_checkpoint=self.resume_from_checkpoint,
            ddp_backend=self.ddp_backend,
            gradient_checkpointing=self.gradient_checkpointing,
            predict_with_generate=self.predict_with_generate,
            # generation_config=generation_config,
            local_rank=get_dist_setting()[1],
            save_only_model=self.save_only_model,
            train_sampler_random=self.train_sampler_random,
            report_to=self.report_to,
            deepspeed=self.deepspeed,
            additional_saved_files=additional_saved_files,
            disable_tqdm=self.disable_tqdm,
            save_on_each_node=self.save_on_each_node,
            acc_strategy=self.acc_strategy,
            save_safetensors=self.save_safetensors,
            logging_first_step=True,
            **kwargs)

        training_args.ddp_find_unused_parameters = self.ddp_find_unused_parameters
        training_args.ddp_broadcast_buffers = self.ddp_broadcast_buffers
        if is_dist() and training_args.ddp_find_unused_parameters is None:
            if self.gradient_checkpointing:
                training_args.ddp_find_unused_parameters = False
            else:
                training_args.ddp_find_unused_parameters = True

        if is_dist() and training_args.ddp_broadcast_buffers is None:
            if self.gradient_checkpointing:
                training_args.ddp_broadcast_buffers = False
            else:
                training_args.ddp_broadcast_buffers = True

        self.training_args = training_args

    def _handle_pai_compat(self) -> None:
        assert is_pai_training_job() is True
        logger.info('Handle pai compat...')
        pai_tensorboard_dir = get_pai_tensorboard_dir()
        if self.logging_dir is None and pai_tensorboard_dir is not None:
            self.logging_dir = pai_tensorboard_dir
            logger.info(f'Setting args.logging_dir: {self.logging_dir}')
        if self.add_output_dir_suffix is None:
            self.add_output_dir_suffix = False
            logger.info(
                f'Setting args.add_output_dir_suffix: {self.add_output_dir_suffix}'
            )


@dataclass
class InferArguments:
    # You can specify the model by either using the model_type or model_id_or_path.
    model_type: Optional[str] = field(
        default=None,
        metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    model_id_or_path: Optional[str] = None
    model_revision: Optional[str] = None

    sft_type: Literal['lora', 'longlora', 'full', 'adalora', 'ia3',
                      'llamapro'] = 'lora'
    template_type: str = field(
        default='AUTO',
        metadata={
            'help':
            f"template_type choices: {list(TEMPLATE_MAPPING.keys()) + ['AUTO']}"
        })
    infer_backend: Literal['AUTO', 'vllm', 'pt'] = 'AUTO'
    ckpt_dir: Optional[str] = field(
        default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'})
    load_args_from_ckpt_dir: bool = True
    load_dataset_config: bool = False
    eval_human: Optional[bool] = None

    seed: int = 42
    dtype: Literal['bf16', 'fp16', 'fp32', 'AUTO'] = 'AUTO'

    dataset: List[str] = field(
        default_factory=list,
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: int = 42
    dataset_test_ratio: float = 0.01
    val_dataset_sample: int = 10  # -1: all dataset
    save_result: bool = True
    system: Optional[str] = None
    max_length: int = -1  # -1: no limit
    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete'
    check_dataset_strategy: Literal['none', 'discard', 'error',
                                    'warning'] = 'none'
    custom_train_dataset_path: List[str] = field(default_factory=list)
    custom_val_dataset_path: List[str] = field(default_factory=list)

    quantization_bit: Literal[0, 4, 8] = 0
    bnb_4bit_comp_dtype: Literal['fp16', 'bf16', 'fp32', 'AUTO'] = 'AUTO'
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True

    max_new_tokens: int = 2048
    do_sample: bool = True
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.7
    repetition_penalty: float = 1.
    num_beams: int = 1
    stop_words: List[str] = None

    # other
    use_flash_attn: Optional[bool] = None
    ignore_args_error: bool = False  # True: notebook compatibility
    stream: bool = True
    merge_lora: bool = False
    merge_device_map: Optional[str] = None
    save_safetensors: bool = True
    overwrite_generation_config: Optional[bool] = None
    verbose: Optional[bool] = None
    # vllm
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    vllm_enable_lora: bool = False
    vllm_max_lora_rank: int = 16
    vllm_lora_modules: List[str] = field(default_factory=list)
    # compatibility. (Deprecated)
    show_dataset_sample: int = 10
    safe_serialization: Optional[bool] = None
    model_cache_dir: Optional[str] = None
    merge_lora_and_save: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.ckpt_dir is not None and not self.check_ckpt_dir_correct(
                self.ckpt_dir):
            logger.warning(
                f'The checkpoint dir {self.ckpt_dir} passed in is invalid, please make sure'
                'the dir contains a `configuration.json` file.')
        handle_compatibility(self)
        handle_path(self)
        logger.info(f'ckpt_dir: {self.ckpt_dir}')
        if self.ckpt_dir is None and self.load_args_from_ckpt_dir:
            self.load_args_from_ckpt_dir = False
            logger.info(
                'Due to `ckpt_dir` being `None`, `load_args_from_ckpt_dir` is set to `False`.'
            )
        if self.load_args_from_ckpt_dir:
            load_from_ckpt_dir(self)
        else:
            assert self.load_dataset_config is False, 'You need to first set `--load_args_from_ckpt_dir true`.'
        set_model_type(self)
        if isinstance(self.dataset, str):
            self.dataset = [self.dataset]
        register_custom_dataset(self)
        check_flash_attn(self)
        handle_generation_config(self)

        self.torch_dtype, _, _ = select_dtype(self)
        if self.template_type == 'AUTO':
            self.template_type = get_default_template_type(self.model_type)
            logger.info(f'Setting template_type: {self.template_type}')
        has_dataset = (
            len(self.dataset) > 0 or len(self.custom_train_dataset_path) > 0
            or len(self.custom_val_dataset_path) > 0)
        if self.eval_human is None:
            if not has_dataset:
                self.eval_human = True
            else:
                self.eval_human = False
            logger.info(f'Setting self.eval_human: {self.eval_human}')
        elif self.eval_human is False and not has_dataset:
            raise ValueError(
                'Please provide the dataset or set `--load_dataset_config true`.'
            )
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self)

        if self.max_length == -1:
            self.max_length = None
        if self.overwrite_generation_config is None:
            if self.ckpt_dir is None:
                self.overwrite_generation_config = False
            else:
                self.overwrite_generation_config = True
            logger.info(
                f'Setting overwrite_generation_config: {self.overwrite_generation_config}'
            )
        if self.ckpt_dir is None:
            self.sft_type = 'full'
        model_info = MODEL_MAPPING[self.model_type]
        support_vllm = model_info.get('support_vllm', False)
        self.vllm_lora_request_list = None
        if self.infer_backend == 'AUTO':
            self.infer_backend = 'pt'
            if is_vllm_available() and support_vllm:
                if ((self.sft_type == 'full'
                     or self.sft_type == 'lora' and self.merge_lora)
                        and self.quantization_bit == 0):
                    self.infer_backend = 'vllm'
                if self.vllm_enable_lora:
                    self.infer_backend = 'vllm'
        if self.infer_backend == 'vllm':
            require_version('vllm')
            assert self.quantization_bit == 0, 'VLLM does not support bnb.'
            if not support_vllm:
                logger.warning(f'vllm not support `{self.model_type}`')
            if self.sft_type == 'lora' and not self.vllm_enable_lora:
                assert self.merge_lora is True, (
                    'To use VLLM, you need to provide the complete weight parameters. '
                    'Please set `--merge_lora true`.')
            if self.vllm_enable_lora:
                self.vllm_lora_modules.append(f'default-lora={self.ckpt_dir}')
                self.vllm_lora_request_list = _parse_vllm_lora_modules(
                    self.vllm_lora_modules)
                logger.info(
                    f'args.vllm_lora_request_list: {self.vllm_lora_request_list}'
                )
        template_info = TEMPLATE_MAPPING[self.template_type]
        if self.num_beams != 1:
            self.stream = False
            logger.info('Setting self.stream: False')
        self.infer_media_type = template_info.get('infer_media_type', 'none')
        if self.merge_device_map is None:
            self.merge_device_map = 'cpu'

    @staticmethod
    def check_ckpt_dir_correct(ckpt_dir) -> bool:
        """Check the checkpoint dir is correct, which means it must contains a `configuration.json` file.
        Args:
            ckpt_dir: The checkpoint dir
        Returns:
            A bool value represents the dir is valid or not.
        """
        if not os.path.exists(ckpt_dir):
            return False
        return os.path.isfile(os.path.join(ckpt_dir, 'configuration.json'))


@dataclass
class AppUIArguments(InferArguments):
    server_name: str = '127.0.0.1'
    server_port: int = 7860
    share: bool = False


@dataclass
class DeployArguments(InferArguments):
    host: str = '127.0.0.1'
    port: int = 8000
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        model_info = MODEL_MAPPING[self.model_type]
        tags = model_info.get('tags', [])
        if 'multi-modal' in tags:
            raise ValueError(
                'Deployment of multimodal models is currently not supported.')


@dataclass
class EvalArguments(InferArguments):

    name: Optional[str] = None

    eval_dataset: List[str] = field(
        default_factory=lambda: ['mmlu', 'ceval', 'gsm8k', 'arc', 'bbh'])

    eval_limit: Optional[int] = None


@dataclass
class ExportArguments(InferArguments):
    # The parameter has been defined in InferArguments.
    # merge_lora: bool = False

    # awq: 4; gptq: 2, 3, 4, 8
    quant_bits: int = 0  # e.g. 4
    quant_method: Literal['awq', 'gptq'] = 'awq'
    quant_n_samples: Optional[int] = None
    quant_seqlen: int = 2048
    quant_device_map: str = 'cpu'  # e.g. 'cpu', 'auto'

    # push to ms hub
    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'SDK token can be found in https://modelscope.cn/my/myaccesstoken'
        })
    hub_private_repo: bool = False
    commit_message: str = 'update files'

    def __post_init__(self):
        if self.merge_device_map is None:
            self.merge_device_map = 'cpu' if self.quant_bits != 0 else 'auto'
        super().__post_init__()
        if len(self.dataset) == 0:
            self.dataset = ['ms-bench-mini']
            logger.info(f'Setting args.dataset: {self.dataset}')
        if self.quant_n_samples is None:
            if self.quant_method == 'awq':
                self.quant_n_samples = 256
            elif self.quant_method == 'gptq':
                self.quant_n_samples = 1024
            else:
                raise ValueError(
                    f'args.quant_n_samples: {self.quant_n_samples}')


@dataclass
class DPOArguments(SftArguments):

    ref_model_type: Optional[str] = field(
        default=None,
        metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})

    max_prompt_length: int = 1024
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: Literal['sigmoid', 'hinge', 'ipo', 'kto_pair'] = 'sigmoid'
    sft_beta: float = 0.1


@dataclass
class RomeArguments(InferArguments):
    rome_request_file: str = field(
        default=None,
        metadata={
            'help':
            'The rome request file, please check the documentation '
            'to get the format'
        })

    def __post_init__(self) -> None:
        handle_compatibility(self)
        handle_path(self)
        set_model_type(self)
        check_flash_attn(self)

        self.torch_dtype, _, _ = select_dtype(self)
        if self.template_type == 'AUTO':
            self.template_type = get_default_template_type(self.model_type)
            logger.info(f'Setting template_type: {self.template_type}')

        if self.max_length == -1:
            self.max_length = None


dtype_mapping_reversed = {v: k for k, v in dtype_mapping.items()}


def select_dtype(
    args: Union[SftArguments, InferArguments]
) -> Tuple[Optional[Dtype], bool, bool]:
    if not is_torch_cuda_available() and not is_torch_npu_available():
        # cpu
        if args.dtype == 'AUTO':
            args.dtype = 'fp32'
            logger.info(f'Setting args.dtype: {args.dtype}')
        assert args.dtype != 'fp16', 'The CPU does not support matrix multiplication with FP16.'
        if args.dtype == 'fp32':
            return torch.float32, False, False
        elif args.dtype == 'bf16':
            return torch.bfloat16, False, True
        else:
            raise ValueError(f'args.dtype: {args.dtype}')
    # cuda, npu
    if args.dtype == 'AUTO' and not is_torch_bf16_gpu_available():
        args.dtype = 'fp16'
    if args.dtype == 'AUTO' and ('int4' in args.model_type
                                 or 'int8' in args.model_type):
        model_torch_dtype = MODEL_MAPPING[args.model_type]['torch_dtype']
        if model_torch_dtype is not None:
            args.dtype = dtype_mapping[model_torch_dtype]
    if args.dtype == 'AUTO':
        if isinstance(args, SftArguments):
            args.dtype = 'bf16'
        else:
            return None, False, False

    torch_dtype = dtype_mapping_reversed[args.dtype]

    assert torch_dtype in {torch.float16, torch.bfloat16, torch.float32}
    if torch_dtype == torch.float16:
        if isinstance(args, SftArguments) and args.sft_type == 'full':
            args.dtype = 'fp32'
            torch_dtype = torch.float32
            logger.warning(
                'Fine-tuning with full parameters does not support fp16, and is prone to NaN. '
                'We will use the fp32 & AMP approach, which consumes approximately twice the memory of bf16.'
            )
            logger.info(f'Setting torch_dtype: {torch_dtype}')
        fp16, bf16 = True, False
    elif torch_dtype == torch.bfloat16:
        support_bf16 = is_torch_bf16_gpu_available()
        if not support_bf16:
            logger.warning(f'support_bf16: {support_bf16}')
        fp16, bf16 = False, True
    else:
        fp16, bf16 = False, False
    return torch_dtype, fp16, bf16


def select_bnb(
    args: Union[SftArguments, InferArguments]
) -> Tuple[Optional[Dtype], bool, bool]:
    if args.bnb_4bit_comp_dtype == 'AUTO':
        args.bnb_4bit_comp_dtype = args.dtype

    if args.bnb_4bit_comp_dtype != 'AUTO':
        bnb_4bit_compute_dtype = dtype_mapping_reversed[
            args.bnb_4bit_comp_dtype]
        assert bnb_4bit_compute_dtype in {
            torch.float16, torch.bfloat16, torch.float32
        }
    else:
        bnb_4bit_compute_dtype = None
    quantization_bit = args.quantization_bit
    if quantization_bit == 4:
        require_version('bitsandbytes')
        load_in_4bit, load_in_8bit = True, False
    elif quantization_bit == 8:
        require_version('bitsandbytes')
        load_in_4bit, load_in_8bit = False, True
    else:
        load_in_4bit, load_in_8bit = False, False

    return bnb_4bit_compute_dtype, load_in_4bit, load_in_8bit


def handle_compatibility(args: Union[SftArguments, InferArguments]) -> None:
    template_type_mapping = {
        'chatglm2-generation': 'chatglm-generation',
        'chatml': 'qwen'
    }
    model_type_mapping = {
        'openbmb-minicpm-2b-sft-chat': 'minicpm-2b-sft-chat',
        'openbmb-minicpm-2b-chat': 'minicpm-2b-chat',
    }
    for k, v in template_type_mapping.items():
        if k == args.template_type:
            args.template_type = v
            break
    for k, v in model_type_mapping.items():
        if k == args.model_type:
            args.model_type = v
            break
    if args.dataset is not None and len(
            args.dataset) == 1 and ',' in args.dataset[0]:
        args.dataset = args.dataset[0].split(',')
    if args.truncation_strategy == 'ignore':
        args.truncation_strategy = 'delete'
    if isinstance(args, InferArguments):
        if args.show_dataset_sample != 10 and args.val_dataset_sample == 10:
            # args.val_dataset_sample is the default value and args.show_dataset_sample is not the default value.
            args.val_dataset_sample = args.show_dataset_sample
        if args.safe_serialization is not None:
            args.save_safetensors = args.safe_serialization
        if args.merge_lora_and_save is not None:
            args.merge_lora = args.merge_lora_and_save
    if isinstance(args, SftArguments):
        if args.only_save_model is not None:
            args.save_only_model = args.only_save_model
        if args.neftune_alpha is not None:
            args.neftune_noise_alpha = args.neftune_alpha
        if args.per_device_train_batch_size is not None:
            args.batch_size = args.per_device_train_batch_size
        if args.per_device_eval_batch_size is not None:
            args.eval_batch_size = args.per_device_eval_batch_size
        if args.deepspeed_config_path is not None:
            args.deepspeed = args.deepspeed_config_path


def set_model_type(args: Union[SftArguments, InferArguments]) -> None:
    # compat with swift<1.7
    if args.model_cache_dir is not None and args.model_id_or_path is None:
        args.model_id_or_path = args.model_cache_dir
        args.model_cache_dir = None

    if args.model_id_or_path is not None:
        model_mapping_reversed = {
            v['model_id_or_path'].lower(): k
            for k, v in MODEL_MAPPING.items()
        }
        model_id_or_path = args.model_id_or_path
        model_id_or_path_lower = model_id_or_path.lower()
        if model_id_or_path_lower not in model_mapping_reversed:
            if (isinstance(args, InferArguments)
                    and 'checkpoint' in model_id_or_path
                    and 'merged' not in model_id_or_path
                    and args.ckpt_dir is None):
                raise ValueError(
                    'Please use `--ckpt_dir vx-xxx/checkpoint-xxx` to use the checkpoint.'
                )
            if args.model_type is None:
                raise ValueError(
                    f"model_id_or_path: '{model_id_or_path}' is not registered. "
                    'Please set `--model_type <model_type>` additionally.')
            assert args.model_cache_dir is None
        else:
            model_type = model_mapping_reversed[model_id_or_path_lower]
            assert args.model_type is None or args.model_type == model_type
            args.model_type = model_type
            logger.info(f'Setting args.model_type: {model_type}')
            if args.model_cache_dir is not None:
                args.model_id_or_path = args.model_cache_dir

    error_msg = f'The model_type you can choose: {list(MODEL_MAPPING.keys())}'
    if args.model_type is None:
        raise ValueError('please setting `--model_type <model_type>`. '
                         + error_msg)
    elif args.model_type not in MODEL_MAPPING:
        raise ValueError(f"model_type: '{args.model_type}' is not registered. "
                         + error_msg)
    model_info = MODEL_MAPPING[args.model_type]
    if args.model_revision is None:
        args.model_revision = model_info['revision']
    else:
        model_info['revision'] = args.model_revision
        logger.info(f"Setting model_info['revision']: {args.model_revision}")
    if args.model_id_or_path is None:
        args.model_id_or_path = model_info['model_id_or_path']
    requires = model_info['requires']
    for require in requires:
        require_version(require)


def prepare_push_ms_hub(args: SftArguments) -> None:
    if not args.push_to_hub:
        return
    if args.hub_model_id is None:
        args.hub_model_id = f'{args.model_type}-{args.sft_type}'
        logger.info(f'Setting hub_model_id: {args.hub_model_id}')

    api = HubApi()
    if args.hub_token is None:
        args.hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
    if args.hub_token is not None:
        api.login(args.hub_token)
    else:
        assert ModelScopeConfig.get_token(
        ) is not None, 'Please enter hub_token'
    logger.info('hub login successful!')


def _check_path(
        k: str, value: Union[str, List[str]],
        check_exist_path_set: Optional[Set[str]]) -> Union[str, List[str]]:
    if isinstance(value, str):
        value = os.path.expanduser(value)
        value = os.path.abspath(value)
        if k in check_exist_path_set and not os.path.exists(value):
            raise FileNotFoundError(f"`{k}`: '{value}'")
    elif isinstance(value, list):
        res = []
        for v in value:
            res.append(_check_path(k, v, check_exist_path_set))
        value = res
    return value


def handle_path(args: Union[SftArguments, InferArguments]) -> None:
    check_exist_path = [
        'ckpt_dir', 'resume_from_checkpoint', 'custom_train_dataset_path',
        'custom_val_dataset_path'
    ]
    if args.model_id_or_path is not None and (
            args.model_id_or_path.startswith('~')
            or args.model_id_or_path.startswith('/')):
        check_exist_path.append('model_id_or_path')
    check_exist_path_set = set(check_exist_path)
    other_path = ['output_dir', 'logging_dir']
    for k in check_exist_path + other_path:
        value = getattr(args, k, None)
        if value is None:
            continue
        value = _check_path(k, value, check_exist_path_set)
        setattr(args, k, value)


def _register_local_dataset(dataset_name: str, train_dataset_path: List[str],
                            val_dataset_path: List[str]) -> None:
    register_dataset(
        dataset_name,
        '_',
        train_dataset_path,
        val_dataset_path,
        get_function=get_custom_dataset,
        exists_ok=True)


def register_custom_dataset(args: Union[SftArguments, InferArguments]) -> None:
    dataset = []
    for d in args.dataset:
        if os.path.exists(d):
            args.custom_train_dataset_path.append(d)
        else:
            dataset.append(d)
    args.dataset = dataset

    for key in ['custom_train_dataset_path', 'custom_val_dataset_path']:
        value = getattr(args, key)
        if isinstance(value, str):
            setattr(args, key, [value])
    if len(args.custom_train_dataset_path) == 0 and len(
            args.custom_val_dataset_path) == 0:
        return

    dataset_name = '_custom_dataset'
    _register_local_dataset(dataset_name, args.custom_train_dataset_path,
                            args.custom_val_dataset_path)
    args.dataset.append(dataset_name)


def load_from_ckpt_dir(args: InferArguments) -> None:
    sft_args_path = os.path.join(args.ckpt_dir, 'sft_args.json')
    if not os.path.exists(sft_args_path):
        logger.info(f'{sft_args_path} not found')
        return
    with open(sft_args_path, 'r', encoding='utf-8') as f:
        sft_args = json.load(f)
    imported_keys = [
        'model_type', 'model_revision', 'sft_type', 'template_type', 'system',
        'quantization_bit', 'bnb_4bit_comp_dtype', 'bnb_4bit_quant_type',
        'bnb_4bit_use_double_quant'
    ]
    if args.load_dataset_config:
        imported_keys += [
            'dataset', 'dataset_seed', 'dataset_test_ratio',
            'check_dataset_strategy', 'custom_train_dataset_path',
            'custom_val_dataset_path'
        ]
    for key in imported_keys:
        if (key in {
                'dataset', 'custom_train_dataset_path',
                'custom_val_dataset_path'
        } and len(getattr(args, key)) > 0):
            continue
        setattr(args, key, sft_args.get(key))


def check_flash_attn(args: Union[SftArguments, InferArguments]) -> None:
    model_info = MODEL_MAPPING[args.model_type]
    support_flash_attn = model_info.get('support_flash_attn', False)
    if args.use_flash_attn and not support_flash_attn:
        logger.warning(f'use_flash_attn: {args.use_flash_attn}, '
                       f'but support_flash_attn: {support_flash_attn}')


def handle_generation_config(
        args: Union[SftArguments, InferArguments]) -> None:
    if args.do_sample is False:
        # fix warning
        args.temperature = 1.
        args.top_p = 1.
        args.top_k = 50
        logger.info(
            'Due to do_sample=False, the following settings are applied: args.temperature: '
            f'{args.temperature}, args.top_p: {args.top_p}, args.top_k: {args.top_k}.'
        )


def handle_dataset_mixture(args: SftArguments,
                           train_dataset: HfDataset) -> None:
    if train_dataset is None:
        return train_dataset
    if args.train_dataset_mix_ratio <= 0 or len(
            args.train_dataset_mix_ds) == 0:
        return train_dataset

    random_state = np.random.RandomState(args.dataset_seed)
    train_dataset_mix_ds = []
    custom_mix_ds = []
    for mix_ds in args.train_dataset_mix_ds:
        if os.path.exists(mix_ds):
            custom_mix_ds.append(mix_ds)
        else:
            train_dataset_mix_ds.append(mix_ds)

    if len(custom_mix_ds) > 0:
        dataset_name = '_custom_mixture'
        _register_local_dataset(dataset_name, custom_mix_ds, [])
        train_dataset_mix_ds.append(dataset_name)
    mix_dataset_sample = int(len(train_dataset) * args.train_dataset_mix_ratio)
    logger.info(f'train_dataset_mix_ds: {train_dataset_mix_ds}')
    logger.info(
        f'len(train_dataset): {len(train_dataset)}, mix_dataset_sample: {mix_dataset_sample}'
    )
    mixed_dataset = get_dataset(
        train_dataset_mix_ds,
        0.0,
        random_state,
        check_dataset_strategy=args.check_dataset_strategy)[0]
    if len(mixed_dataset) < mix_dataset_sample:
        logger.warn(
            f'The length of dataset used for mixin: {train_dataset_mix_ds} are '
            'lesser than the ratio required by the `train_dataset_mix_ratio` '
            f'argument: {args.train_dataset_mix_ratio}. '
            f'the actual ratio is: {len(mixed_dataset)/len(train_dataset):.6}.'
        )
    else:
        train_idxs = random_state.permutation(mix_dataset_sample)
        mixed_dataset = mixed_dataset.select(train_idxs)
    return concatenate_datasets([train_dataset, mixed_dataset])


def swift_to_peft_format(lora_checkpoint_path: str) -> str:
    if 'default' in os.listdir(lora_checkpoint_path):  # swift_backend
        new_lora_checkpoint_path = f'{lora_checkpoint_path}-peft'
        Swift.save_to_peft_format(lora_checkpoint_path,
                                  new_lora_checkpoint_path)
        lora_checkpoint_path = new_lora_checkpoint_path
        logger.info('Converting the swift format checkpoint to peft format, '
                    f"and saving it to: '{new_lora_checkpoint_path}'")
    else:
        logger.info('The format of the checkpoint is already in peft format.')
    return lora_checkpoint_path


def _parse_vllm_lora_modules(
        vllm_lora_modules: List[str]) -> List['LoRARequest']:
    try:
        from .vllm_utils import LoRARequest
    except ImportError:
        logger.warning(
            'The current version of VLLM does not support `enable_lora`. Please upgrade VLLM.'
        )
        raise
    lora_request_list = []
    for i, vllm_lora_module in enumerate(vllm_lora_modules):
        lora_name, lora_local_path = vllm_lora_module.split('=')
        lora_local_path = swift_to_peft_format(lora_local_path)
        lora_request_list.append(
            LoRARequest(lora_name, i + 1, lora_local_path))
    return lora_request_list
