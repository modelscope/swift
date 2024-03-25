# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from dvlab-research/LongLoRA.
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import torch.nn as nn

from swift import LoRA, LoRAConfig, SwiftOutput
from swift.tuners.lora import lora_state_dict, mark_lora_as_trainable
from swift.tuners.lora_layers import LoraModel


class LongLoRAModelType:
    LLAMA = 'llama'


@dataclass
class LongLoRAConfig(LoRAConfig):
    """
    The Config for the LongLoRA adapter.
    LongLoRA:[Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)
    This adapter uses S2-attention to shorten the attention window for long context training scenarios.
    Args:
        embedder_and_normalizer: LongLoRA allows the embedder and normalizer to be trainable, this parameter specifies
            the names of the embedders and normalizers.
        model_type: The model type, now support llama only
        group_size_ratio: The group size window ratio of the sequence length.
            Note: The sequence length should be split to smaller sequences by the ratio.
    """

    embedder_and_normalizer: Union[str, List[str], Tuple[str]] = field(
        default=('embed', 'norm'),
        metadata={
            'help':
            'The names of embedder and normalizer, regex format if is a str, else will match with sub sequences'
        })

    model_type: str = field(
        default=None,
        metadata={
            'help': 'The model type, now only support `llama` structure.'
        })

    group_size_ratio: float = field(
        default=0.25, metadata={'help': 'The S2 attention group ratio'})

    def __post_init__(self):
        from swift.tuners.mapping import SwiftTuners
        self.swift_type = SwiftTuners.LONGLORA


class LongLoRA(LoRA):

    @staticmethod
    def prepare_model(model: nn.Module, config: LongLoRAConfig,
                      adapter_name: str):
        """Prepare a model with `LongLoRAConfig`"""
        LoraModel(model, config, adapter_name)

        def state_dict_callback(state_dict, adapter_name):
            _state_dict = lora_state_dict(state_dict, adapter_name,
                                          config.bias)
            for name, value in state_dict.items():
                if isinstance(config.embedder_and_normalizer, str):
                    target_module_found = re.fullmatch(
                        config.embedder_and_normalizer, name)
                else:
                    target_module_found = any(
                        target_key in name
                        for target_key in config.embedder_and_normalizer)
                if target_module_found and name not in _state_dict:  # noqa
                    _state_dict[name] = value
            return _state_dict

        def mark_trainable_callback(model):
            mark_lora_as_trainable(model, adapter_name, config.bias)
            mark_embedding_normalizer_as_trainable(
                model, config.embedder_and_normalizer)

        if config.model_type == LongLoRAModelType.LLAMA:
            from .llama import replace_llama_attn
            replace_llama_attn(model)
            # only support code base from transformers
            model.config.group_size_ratio = config.group_size_ratio

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback)


def mark_embedding_normalizer_as_trainable(
        model: nn.Module, extra_parameters: Union[str, List[str],
                                                  Tuple[str]]) -> None:
    for name, sub_module in model.named_parameters():
        if isinstance(extra_parameters, str):
            target_module_found = re.fullmatch(extra_parameters, name)
        else:
            target_module_found = any(target_key in name
                                      for target_key in extra_parameters)
        if target_module_found:  # noqa
            sub_module.requires_grad = True
