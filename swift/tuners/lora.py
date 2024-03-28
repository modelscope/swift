# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
from dataclasses import asdict, dataclass, field
from functools import reduce

import torch
from packaging import version
from transformers import Trainer

from .lora_layers import *  # noqa
from .utils import SwiftAdapter, SwiftConfig, SwiftOutput, set_adapter

logger = get_logger()


@dataclass
class LoRAConfig(LoraConfig, SwiftConfig):
    """
    The configuration class for the loRA module.

    Args:
        use_qa_lora(bool): Use
            QA-LoRA:[Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717)
            instead of LoRA. QA-LoRA only supports AutoGPTQ quantized models.
        lora_dtype(str): The dtype for all lora modules, supported values are `fp32`, `fp16`, `bf16`.
            Default value is `None`, which means follow the dtype of original module's weight.
        lr_ratio(float): The lr_ratio argument for [LoRA+](https://arxiv.org/abs/2402.12354)
    """

    use_qa_lora: bool = field(
        default=False,
        metadata={
            'help':
            'Use [qa-lora](https://github.com/yuhuixu1993/qa-lora) or not'
        })

    use_merged_linear: bool = field(
        default=False, metadata={'help': 'Use merged Linear'})

    enable_lora: List[bool] = field(
        default=None,
        metadata={
            'help':
            'The modules need to be turned on when using the merged linear layer'
        })

    lora_dtype: str = field(
        default=None,
        metadata={
            'help':
            'The lora dtype, default None means following the original layer\'s dtype'
        })

    lorap_lr_ratio: float = field(
        default=2.0**4, metadata={'help': 'The lr ratio of lora_B in lora+'})

    lorap_emb_lr: float = field(
        default=1e-6, metadata={'help': 'The lr for embedding in lora+'})

    def __post_init__(self):
        super().__post_init__()
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.LORA

    def can_be_saved_to_peft(self) -> bool:
        if self.use_qa_lora or self.use_merged_linear:
            logger.warn(
                'QA-LoRA and MergedLinear cannot be saved to peft format')
            return False
        return True

    def to_peft_config(self) -> LoraConfig:
        _dict = asdict(self)
        _dict.pop('use_qa_lora', None)
        _dict.pop('enable_lora', None)
        _dict.pop('lora_dtype', None)
        _dict.pop('use_merged_linear', None)
        _dict['peft_type'] = _dict['swift_type']
        _dict.pop('swift_type', None)
        _dict.pop('lr_ratio', None)
        _dict.pop('model_key_mapping', None)
        return LoraConfig(**_dict)


class LoRA(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: LoRAConfig, adapter_name: str):
        assert version.parse(peft.__version__) < version.parse(
            '0.10.0'), 'The swift LoRA adapter only supports peft<0.10.0'

        if config.use_qa_lora:
            auto_gptq_config = get_quantization_config(model, method='gptq')
            if auto_gptq_config:
                config.group_size = getattr(auto_gptq_config, 'group_size',
                                            None)
        LoraModel(model, config, adapter_name)

        def state_dict_callback(state_dict, adapter_name, cfg=None):
            return lora_state_dict(state_dict, adapter_name,
                                   cfg.bias if cfg else config.bias)

        def mark_trainable_callback(model, cfg=None):
            mark_lora_as_trainable(model, adapter_name,
                                   cfg.bias if cfg else config.bias)

        def optimizer_group_callback(model, **defaults):
            if config.lorap_lr_ratio is None:
                return None, None

            def get_module(name):
                parent_idx = 2 if 'lora' in name else 1
                module_names = name.split(sep='.')[:-parent_idx]
                module = reduce(getattr, module_names, model)
                return module

            all_params = set()
            param_groups = {
                'groupA': {},
                'groupB': {},
                'groupB_no_decay': {},
                'embedding': {},
            }

            decay_parameters = Trainer.get_decay_parameter_names(None, model)
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                module = get_module(name)
                if isinstance(module, Embedding):
                    param_groups['embedding'][name] = param
                elif 'lora_B' in name or param.ndim == 1:
                    if name in decay_parameters:
                        param_groups['groupB'][name] = param
                    else:
                        param_groups['groupB_no_decay'][name] = param
                else:
                    param_groups['groupA'][name] = param
                all_params.add(name)

            lr = defaults['lr']
            weight_decay = defaults.get('weight_decay', 0.0)

            param_groups = [
                {
                    'params': list(param_groups['groupA'].values()),
                    'weight_decay': weight_decay,
                    'lr': lr,
                },
                {
                    'params': list(param_groups['embedding'].values()),
                    'weight_decay': weight_decay,
                    'lr': config.lorap_emb_lr,
                },
                {
                    'params': list(param_groups['groupB'].values()),
                    'weight_decay': weight_decay,
                    'lr': lr * config.lorap_lr_ratio,
                },
                {
                    'params': list(param_groups['groupB_no_decay'].values()),
                    'weight_decay': 0.0,
                    'lr': lr * config.lorap_lr_ratio,
                },
            ]
            return all_params, param_groups

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback, optimizer_group_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module,
                         adapter_name: str,
                         activate: bool,
                         offload: str = None):
        set_adapter(module, adapter_name, activate, offload)
        for sub_module in module.modules():
            if isinstance(sub_module, (LoraLayer, LoRALayer)):
                sub_module.set_activation(adapter_name, activate)
                if hasattr(sub_module, 'save_memory'):
                    sub_module.save_memory(adapter_name, activate, offload)

    @staticmethod
    def unpatch_lora(model, config: LoRAConfig, adapter_name: str):
        """Unpatch lora modules and merge the weights to original modules.

        LoRA constructs an additional layer with low-rank decomposition matrices of the weights in the network.
        'LoRA: Low-Rank Adaptation of Large Language Models' by Hu et al.(2021)
        See https://arxiv.org/abs/2106.09685

        Args:
            model(`torch.nn.Module`): The model called with `tune` function.
            config(`LoRAConfig`): The `LoRAConfig` to use. Deprecated
            adapter_name(`str`): The adapter name
        """
        if not config.use_merged_linear:
            if version.parse(peft.__version__) < version.parse('0.6.3'):
                logger.info('All adapters will be merged.')
                LoraModel(model, None, '').merge_and_unload()
            else:
                LoraModel(model, None,
                          '').merge_and_unload(adapter_names=[adapter_name])
        else:
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, MergedLinear):
                    sub_module.merge()
                    parent = model.get_submodule('.'.join(
                        name.split('.')[:-1]))
                    target_name = name.split('.')[-1]
                    setattr(parent, target_name, sub_module.base_layer)
