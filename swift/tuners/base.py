# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.
import os
import re
import shutil
from copy import copy
from functools import partial
from inspect import Parameter, Signature, signature
from types import MethodType
from typing import Dict, List, Literal, Optional, Union

import json
import torch
from peft.utils import CONFIG_NAME
from peft.utils.other import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
from torch import nn
from transformers import Trainer

from swift import SwiftTuners
from swift.hub.snapshot_download import snapshot_download
from swift.utils.constants import DEFAULT_ADAPTER, SWIFT_TYPE_KEY
from swift.utils.logger import get_logger
from .. import PeftConfig, PeftModel, get_peft_model
from .utils import SwiftConfig, SwiftOutput

logger = get_logger()


class SwiftModel(nn.Module):
    """The Swift wrapper model.

    Args:
        model (`Union[nn.Module, 'SwiftModel']`) A module to be tuned by Swift.
        config (`Union[SwiftConfig, Dict[str, SwiftConfig]]`) A config or a dict of {adapter_name: SwiftConfig}.
            If it's a config class, the adapter_name will be `default`
        extra_state_keys (`List[str]`, `optional`) A list of regex to match the extra state keys to be saved.
        inference_mode (bool, `optional`): Load model at inference mode, default False.
    """

    EXTRA_STATE_DIR = 'extra_states'

    def __init__(self,
                 model: Union[nn.Module, 'SwiftModel'],
                 config: Union[SwiftConfig, Dict[str, SwiftConfig]],
                 extra_state_keys: List[str] = None,
                 inference_mode: bool = False,
                 **kwargs):
        super().__init__()
        self.adapters = {}
        self.active_adapters = set()
        if isinstance(model, SwiftModel):
            self.adapters = model.adapters
            extra_state_keys = extra_state_keys or []
            extra_state_keys.extend(model.extra_state_keys)
            self.active_adapters = model.active_adapters
            model = model.base_model

        new_adapters = []
        if isinstance(config, SwiftConfig):
            if DEFAULT_ADAPTER not in self.adapters:
                self.adapters[DEFAULT_ADAPTER] = self._prepare_model(
                    model, config, DEFAULT_ADAPTER)
                new_adapters.append(DEFAULT_ADAPTER)
            else:
                logger.warn(
                    f'Adapter {DEFAULT_ADAPTER} has been patched, skip.')
        elif isinstance(config, dict):
            assert (all(isinstance(c, SwiftConfig) for c in config.values()))
            for adapter_name, _config in config.items():
                if adapter_name not in self.adapters:
                    self.adapters[adapter_name] = self._prepare_model(
                        model, _config, adapter_name)
                    new_adapters.append(adapter_name)
                else:
                    logger.warn(
                        f'Adapter {adapter_name} has been patched, skip.')
        self.base_model = model

        self.extra_state_keys = extra_state_keys or []
        self.has_additional_modules = any(
            [c.config.has_additional_modules for c in self.adapters.values()])

        def forward(self, *args, **kwargs):
            return self.base_model(*args, **kwargs)

        _parameters = [Parameter('self', Parameter.POSITIONAL_ONLY)]
        _parameters += list(
            signature(self.base_model.forward).parameters.values())
        forward.__signature__ = Signature(_parameters)
        self.forward = MethodType(forward, self)
        for adapter_name in new_adapters:
            self.activate_adapter(adapter_name)

        if inference_mode:
            self.eval()
        else:
            for key, output in self.adapters.items():
                if key in new_adapters:
                    output.mark_trainable_callback(model)
            if self.extra_state_keys:
                for n, p in model.named_parameters():
                    if any(
                            re.fullmatch(extra_key, n)
                            for extra_key in self.extra_state_keys):
                        p.requires_grad = True

    @property
    def model(self):
        return self.base_model

    def load_state_dict(self,
                        state_dict,
                        strict=True,
                        adapter_name: str = None):
        if adapter_name is not None:
            output = self.adapters[adapter_name]
            if getattr(output.config, 'modules_to_save', None):
                for key, value in copy(state_dict).items():
                    for module_name in output.config.modules_to_save:
                        if module_name in key:
                            state_dict.pop(key)
                            key = key.replace(
                                module_name,
                                f'{module_name}.modules_to_save.{adapter_name}'
                            )
                            break
                    state_dict[key] = value

            for key, value in copy(state_dict).items():
                if key.startswith('base_model.model.'):
                    state_dict.pop(key, None)
                    key = key[len('base_model.model.'):]
                if f'lora_A.{adapter_name}.' not in key and 'lora_A' in key:
                    state_dict.pop(key, None)
                    key = key.replace('lora_A.', f'lora_A.{adapter_name}.')
                if f'lora_B.{adapter_name}.' not in key and 'lora_B' in key:
                    state_dict.pop(key, None)
                    key = key.replace('lora_B.', f'lora_B.{adapter_name}.')
                if f'lora_embedding_A.{adapter_name}.' not in key and 'lora_embedding_A' in key:
                    state_dict.pop(key, None)
                    key = key.replace('lora_embedding_A.',
                                      f'lora_embedding_A.{adapter_name}.')
                if f'lora_embedding_B.{adapter_name}.' not in key and 'lora_embedding_B' in key:
                    state_dict.pop(key, None)
                    key = key.replace('lora_embedding_B.',
                                      f'lora_embedding_B.{adapter_name}.')
                state_dict[key] = value

        incompatible_keys = self.base_model.load_state_dict(state_dict, False)
        if incompatible_keys and len(incompatible_keys[1]) > 0:
            logger.error(
                f'Load state dict with unexpected keys: {incompatible_keys[1]}'
            )

    def state_dict(self,
                   *args,
                   destination=None,
                   prefix='',
                   keep_vars=False,
                   adapter_name: str = None,
                   peft_format: bool = False,
                   **kwargs):
        """
        Args:
            destination (`dict`, `optional`): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (`str`, `optional`): a prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (`bool`, `optional`): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.
            adapter_name (`str`, `optional`): The name of the adapter's parameters to be saved,
                `None` input will save all adapters.
            peft_format (`bool`, `optional`): Save with peft format (extra `base_model.model.` prefix)
            **kwargs:
                save_adapter(`bool`): Save adapters or not, default True
                save_extra_states(`bool`): Save extra states or not, default True
        Returns:
            The state dict to be saved.
        """
        state_dict = kwargs.get('state_dict')
        if state_dict is None:
            state_dict = self.base_model.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict = {
            key[len('base_model.'):] if key.startswith('base_model.') else key:
            value
            for key, value in state_dict.items()
        }
        if not self.has_additional_modules:
            return state_dict

        state_dicts = {}
        if kwargs.get('save_adapter', True):
            for name, output in self.adapters.items():
                if (adapter_name == name or adapter_name is None
                    ) and output.config.has_additional_modules:  # noqa
                    state_dicts.update(
                        output.state_dict_callback(state_dict, name))
                    modules_to_save_names = [
                        sub_name
                        for sub_name, _ in self.base_model.named_parameters()
                        if f'modules_to_save.{name}' in sub_name
                    ]
                    for module_name in modules_to_save_names:
                        if f'modules_to_save.{name}' in module_name:
                            state_dicts[module_name.replace(
                                f'modules_to_save.{name}.',
                                '')] = state_dict[module_name]
        if kwargs.get('save_extra_states', True):
            state_dicts.update({
                k: v
                for k, v in state_dict.items() if any(
                    re.fullmatch(extra_key, k)
                    for extra_key in self.extra_state_keys)
            })
        if peft_format:
            new_state_dict = {}
            for key, value in state_dicts.items():
                if not key.startswith('base_model.model.'):
                    key = 'base_model.model.' + key
                key = key.replace(f'lora_A.{adapter_name}.', 'lora_A.')
                key = key.replace(f'lora_B.{adapter_name}.', 'lora_B.')
                key = key.replace(f'lora_embedding_A.{adapter_name}.',
                                  'lora_embedding_A.')
                key = key.replace(f'lora_embedding_B.{adapter_name}.',
                                  'lora_embedding_B.')
                new_state_dict[key] = value
            state_dicts = new_state_dict
        return state_dicts

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    @staticmethod
    def load_state_file(path, device: Optional[str] = None):
        """Load a state dict file by the input path.

        Args:
            path: The local dir to load the state file.

        Returns:
            The state dict.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
            filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
            from safetensors.torch import load_file as safe_load_file
            return safe_load_file(filename, device=device)
        elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
            filename = os.path.join(path, WEIGHTS_NAME)
            return torch.load(filename, map_location=device)
        return None

    def create_optimizer_param_groups(self, **defaults):
        all_param_names = set()
        param_groups = []
        for output in self.adapters.values():
            if output.optimizer_group_callback:
                param_names, param_group = output.optimizer_group_callback(
                    self.model, **defaults)
                if param_names and all_param_names & param_names:
                    raise ValueError(
                        'Cannot set one parameter to different param groups')
                if param_names and param_group:
                    all_param_names.update(param_names)
                    param_groups.extend(param_group)

        decay_parameters = Trainer.get_decay_parameter_names(None, self.model)
        param_groups.extend([
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if (n in decay_parameters and n not in all_param_names
                        and p.requires_grad)
                ],
                'weight_decay':
                defaults['weight_decay'],
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if (n not in decay_parameters and n not in all_param_names
                        and p.requires_grad)
                ],
                'weight_decay':
                0.0,
            },
        ])

        return param_groups

    @classmethod
    def from_pretrained(cls,
                        model: Union[nn.Module, 'SwiftModel'],
                        model_id: str = None,
                        adapter_name: Union[str, List[str], Dict[str,
                                                                 str]] = None,
                        inference_mode: bool = False,
                        revision: str = None,
                        **kwargs):
        """Load a set of tuners and corresponding weights by a model_id.

        Args:
            model (`Union[torch.nn.Module, 'SwiftModel']`): The model to be tuned,
                if the model is already a `SwiftModel` it will be un-wrapped and re-wrapped..
            model_id (`str`): The model_id or a local model dir of tuners to use to tune the model.
            adapter_name (`Union[str, List[str], Dict[str, str]]`): The adapter_names saved in the model repo to load.
                Default `None`, means load all tuners saved in the model_id
            inference_mode (`bool`): Use in the inference mode or not.
            revision (`str`): The model revision to use.
            **kwargs:
                extra_state_keys (`List[str]`, `optional`) A list of regex to match the extra state keys to be saved.
                Other parameters will be passed to the device_map.
        Returns:
            The `SwiftModel` instance.
        """
        adapters = {}
        model_dir = model_id
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir, revision=revision)
        if os.path.isfile(model_dir):
            raise ValueError(
                f'Please pass in a local dir or a model id, not a local file: {model_dir}'
            )
        extra_state_keys = kwargs.pop('extra_state_keys', None)
        if extra_state_keys is None and os.path.isfile(
                os.path.join(model_dir, cls.EXTRA_STATE_DIR, CONFIG_NAME)):
            with open(
                    os.path.join(model_dir, cls.EXTRA_STATE_DIR, CONFIG_NAME),
                    'r') as file:
                _json = json.load(file)
                extra_state_keys = _json.get('extra_state_keys')
        if adapter_name is None:
            adapter_name = [
                sub_dir for sub_dir in os.listdir(model_dir) if
                os.path.isfile(os.path.join(model_dir, sub_dir, CONFIG_NAME))
                and sub_dir != cls.EXTRA_STATE_DIR
            ]
        for _name in adapter_name if isinstance(adapter_name,
                                                list) else [adapter_name] \
                if isinstance(adapter_name, str) else adapter_name.keys():
            sub_folder = os.path.join(model_dir, _name)
            config_file = os.path.join(sub_folder, CONFIG_NAME)

            if not os.path.isfile(config_file):
                logger.warning(f'{_name} is not a valid tuner')
                continue

            with open(config_file, 'r') as file:
                json_object = json.load(file)

            if SWIFT_TYPE_KEY not in json_object:
                raise ValueError('Mixed using with peft is not allowed now.')
            else:
                key = _name if not isinstance(adapter_name,
                                              dict) else adapter_name[_name]
                adapters[key] = SwiftConfig.from_pretrained(sub_folder)

        self = SwiftModel(model, adapters, extra_state_keys, inference_mode,
                          **kwargs)
        for _name in adapter_name if isinstance(adapter_name,
                                                list) else [adapter_name] \
                if isinstance(adapter_name, str) else adapter_name.keys():
            sub_folder = os.path.join(model_dir, _name)
            state_dict = cls.load_state_file(sub_folder)
            _adapter = _name if not isinstance(adapter_name,
                                               dict) else adapter_name[_name]
            if state_dict is not None:
                model_is_qlora = len([
                    k for k in self.state_dict().keys()
                    if k.endswith(f'.lora_A.{_adapter}.weight')
                    or k.endswith(f'.lora_B.{_adapter}.weight')
                ])
                if not model_is_qlora:
                    # model is lora, state_dict: qlora->lora
                    state_dict = {
                        k[:-len(f'.{_name}.weight') if k.
                          endswith(f'.lora_A.{_name}.weight') or k.
                          endswith(f'.lora_B.{_name}.weight') else None]: v
                        for k, v in state_dict.items()
                    }
                if any(['loramodule' in key for key in state_dict]):
                    # Compatible with old checkpoints before ms-swift:1.5.0
                    state_dict = {
                        key.replace(f'loramodule_{_name}.lora_A', 'lora_A')
                        if f'loramodule_{_name}.lora_A.{_name}' in key else
                        key.replace(f'loramodule_{_name}.lora_A',
                                    f'lora_A.{_name}.weight'): value
                        for key, value in state_dict.items()
                    }
                    state_dict = {
                        key.replace(f'loramodule_{_name}.lora_B', 'lora_B')
                        if f'loramodule_{_name}.lora_B.{_name}' in key else
                        key.replace(f'loramodule_{_name}.lora_B',
                                    f'lora_B.{_name}.weight'): value
                        for key, value in state_dict.items()
                    }
                if isinstance(adapter_name, dict):
                    # TODO this logic is fragile! replace `_name` may cause other parts replaced
                    state_dict = {
                        key.replace(_name, adapter_name[_name]): value
                        for key, value in state_dict.items()
                    }
                self.load_state_dict(state_dict, adapter_name=_adapter)
        state_dict = cls.load_state_file(
            os.path.join(model_dir, self.EXTRA_STATE_DIR))
        if state_dict is not None:
            self.load_state_dict(state_dict)
        return self

    @classmethod
    def _prepare_model(
        cls,
        model: nn.Module,
        config: SwiftConfig,
        adapter_name: str,
    ):
        assert (hasattr(config, SWIFT_TYPE_KEY))
        from .mapping import SWIFT_MAPPING

        adatper_cls = SWIFT_MAPPING[config.swift_type][1]
        if adatper_cls.has_additional_modules() and not getattr(
                model, 'model_frozen', False):
            for _, p in model.named_parameters():
                p.requires_grad = False
            model.model_frozen = True
        config.has_additional_modules = adatper_cls.has_additional_modules()
        return adatper_cls.prepare_model(model, config, adapter_name)

    def create_or_update_model_card(self, output_dir: str):
        """
        Updates or create the model card.
        """
        if not os.path.exists(os.path.join(output_dir, 'README.md')):
            lines = []
        else:
            with open(os.path.join(output_dir, 'README.md'), 'r') as f:
                lines = f.readlines()

        quantization_config = None
        if hasattr(self.base_model, 'config') and hasattr(
                self.base_model.config, 'quantization_config'):
            quantization_config = self.base_model.config.quantization_config.to_dict(
            )
        training_config_text = ''
        # Adds quantization information if it was used
        if quantization_config is not None:
            training_config_text += '\nThe following `bitsandbytes` quantization config was used during training:\n'
            training_config_text += '\n'.join([
                f'- {name}: {value}'
                for name, value in quantization_config.items()
            ])
            training_config_text += '\n'

        training_procedure_heading = '## Training procedure\n'
        if training_procedure_heading in lines:
            lines.insert(
                lines.index(training_procedure_heading) + 2,
                training_config_text)
        else:
            lines.append(
                f'{training_procedure_heading}\n{training_config_text}')

        framework_block_heading = '### Framework versions\n'
        from swift.version import __version__
        if framework_block_heading in lines:
            lines.insert(
                lines.index(framework_block_heading) + 2,
                f'- SWIFT {__version__}\n')
        else:
            lines.append(
                f'{framework_block_heading}\n\n- SWIFT {__version__}\n')

        base_model_heading = '### Base model information\n'
        lines.append(
            f'{base_model_heading}\n\n- BaseModel Class {self.base_model.__class__.__name__}\n'
        )

        # write the lines back to README.md
        with open(os.path.join(output_dir, 'README.md'), 'w') as f:
            f.writelines(lines)

    def add_weighted_adapter(
        self,
        adapters,
        weights,
        adapter_name,
        combination_type='svd',
        svd_rank=None,
        svd_clamp=None,
        svd_full_matrices=True,
        svd_driver=None,
        density=None,
        majority_sign_method: Literal['total', 'frequency'] = 'total',
    ):
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                The merging type can be one of [`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`,
                `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]. When using the `cat`
                combination_type, the rank of the resulting adapter is equal to the sum of all adapters ranks (the
                mixed adapter may be too big and result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
            density (`float`, *optional*):
                Value between 0 and 1. 0 means all values are pruned and 1 means no values are pruned. Should be used
                with [`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`,
                `magnintude_prune`, `magnitude_prune_svd`]
            majority_sign_method (`str`):
                The method, should be one of ["total", "frequency"], to use to get the magnitude of the sign values.
                Should be used with [`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]
        """
        from swift.tuners.lora import LoraModel
        lora_model = LoraModel(self.model, None, '')
        lora_model.peft_config = {
            key: value.config
            for key, value in self.adapters.items()
        }
        from peft.tuners.lora import LoraLayer
        lora_model.targeted_module_names = [
            key for key, value in self.model.named_modules()
            if isinstance(value, LoraLayer)
        ]
        lora_model.active_adapter = self.active_adapters
        lora_model.add_weighted_adapter(
            adapters=adapters,
            weights=weights,
            adapter_name=adapter_name,
            combination_type=combination_type,
            svd_rank=svd_rank,
            svd_clamp=svd_clamp,
            svd_full_matrices=svd_full_matrices,
            svd_driver=svd_driver,
            density=density,
            majority_sign_method=majority_sign_method,
        )

        def state_dict_callback(state_dict, adapter_name, cfg):
            from swift.tuners.lora_layers import lora_state_dict
            return lora_state_dict(state_dict, adapter_name, cfg.bias)

        def mark_trainable_callback(model, cfg):
            from swift.tuners.lora_layers import mark_lora_as_trainable
            mark_lora_as_trainable(model, adapter_name, cfg.bias)

        cfg = lora_model.peft_config[adapter_name]
        cfg.has_additional_modules = True
        self.adapters[adapter_name] = SwiftOutput(
            config=cfg,
            state_dict_callback=partial(state_dict_callback, cfg=cfg),
            mark_trainable_callback=partial(mark_trainable_callback, cfg=cfg),
            optimizer_group_callback=None,
        )

        self.set_active_adapters(adapter_name)

    def save_pretrained(self,
                        save_directory: str,
                        safe_serialization: bool = False,
                        adapter_name: Union[str, List[str]] = None,
                        **kwargs):
        """Save the adapters to a local directory.

        Args:
            save_directory (`str`): The directory to use.
            safe_serialization (`bool`): Use safe tensors to save the weights, default False.
            adapter_name(`Union[str, List[str]]`): The adapters to be saved, default is `None` to save all.
        """
        peft_format = kwargs.pop('peft_format', False)
        if os.path.isfile(save_directory):
            raise ValueError(
                f'Provided path ({save_directory}) should be a directory, not a file'
            )
        os.makedirs(save_directory, exist_ok=True)
        if not self.has_additional_modules:
            if hasattr(self.base_model, 'save_pretrained'):
                self.base_model.save_pretrained(
                    save_directory, safe_serialization=safe_serialization)
            else:
                self._save_state_dict(self.base_model.state_dict(),
                                      save_directory, safe_serialization)
                self.create_or_update_model_card(save_directory)
        else:
            self.create_or_update_model_card(save_directory)

        adapter_names = adapter_name if isinstance(
            adapter_name, list) or adapter_name is None else [adapter_name]

        state_dict_kwargs = {}
        state_dict = kwargs.get('state_dict')
        if state_dict is not None:
            state_dict_kwargs['state_dict'] = kwargs['state_dict']
        for adapter_name, output in self.adapters.items():
            if adapter_names is not None and adapter_name not in adapter_names:
                continue
            save_to_peft = peft_format and output.config.swift_type == SwiftTuners.LORA
            save_to_peft = save_to_peft and output.config.can_be_saved_to_peft(
            )
            if peft_format and not save_to_peft:
                logger.error(
                    'You are using additional lora parameters, which is not compatible with peft,'
                    'which is unable to save to peft format.')
            # save only the trainable weights
            output_state_dict = self.state_dict(
                adapter_name=adapter_name,
                save_extra_states=False,
                peft_format=save_to_peft,
                **state_dict_kwargs)
            output_dir = os.path.join(
                save_directory, adapter_name
            ) if adapter_name != 'default' or not save_to_peft else save_directory
            os.makedirs(output_dir, exist_ok=True)
            if output_state_dict and output.config.has_additional_modules:
                self._save_state_dict(output_state_dict, output_dir,
                                      safe_serialization)
            if save_to_peft:
                config = output.config.to_peft_config()
                config.save_pretrained(output_dir)
            else:
                output.config.save_pretrained(output_dir)

        output_state_dict = self.state_dict(
            save_extra_states=True, save_adapter=False, **state_dict_kwargs)
        if len(output_state_dict) > 0:
            if self.has_additional_modules:
                os.makedirs(
                    os.path.join(save_directory, self.EXTRA_STATE_DIR),
                    exist_ok=True)
                self._save_state_dict(
                    output_state_dict,
                    os.path.join(save_directory, self.EXTRA_STATE_DIR),
                    safe_serialization)
                with open(
                        os.path.join(save_directory, self.EXTRA_STATE_DIR,
                                     CONFIG_NAME), 'w') as file:
                    json.dump({'extra_state_keys': self.extra_state_keys},
                              file)
            else:
                logger.error(
                    'Full parameter training, save_extra_states will be ignored'
                )

        if not os.path.exists(
                os.path.join(save_directory, 'configuration.json')):
            with open(os.path.join(save_directory, 'configuration.json'),
                      'w') as f:
                f.write('{}')

    @staticmethod
    def _save_state_dict(output_state_dict, save_directory,
                         safe_serialization):
        if safe_serialization:
            from safetensors.torch import save_file as safe_save_file
            safe_save_file(
                output_state_dict,
                os.path.join(save_directory, SAFETENSORS_WEIGHTS_NAME),
                metadata={'format': 'pt'})
        else:
            torch.save(output_state_dict,
                       os.path.join(save_directory, WEIGHTS_NAME))

    def set_active_adapters(self,
                            adapter_names: Union[List[str], str],
                            offload: str = None):
        """Set activated adapters

        Args:
            adapter_names(`Union[List[str], str]`): The adapters needed to be activated
            offload(`str`): Whether to offload the deactivated ones to `cpu` or `meta` device
        """
        if not adapter_names:
            adapter_names = []

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        adapter_names = set(adapter_names)
        for adapter_name in (adapter_names & set(self.adapters.keys())):
            self.activate_adapter(adapter_name)

        for adapter_name in (set(self.adapters.keys()) - adapter_names):
            self.deactivate_adapter(adapter_name, offload)

        self.active_adapters = (adapter_names & set(self.adapters.keys()))

    def activate_adapter(self, adapter_name: str):
        """Activate one adapter

        Args:
            adapter_name(`str`): The adapter needed to be activated
        """
        if adapter_name not in self.adapters:
            logger.warning(
                f'{adapter_name} not in adapters: {self.adapters.keys()}')
            return

        from .mapping import SWIFT_MAPPING
        SWIFT_MAPPING[self.adapters[adapter_name].config.swift_type][1]\
            .activate_adapter(self.base_model, adapter_name, True)
        self.active_adapters = self.active_adapters | {adapter_name}

    def deactivate_adapter(self, adapter_name: str, offload: str = None):
        """Deactivate one adapter

        Args:
            adapter_name(`str`): The adapter needed to be activated
            offload(`str`): Whether to offload to `cpu` or `meta` device
        """
        if adapter_name not in self.adapters:
            logger.warning(
                f'{adapter_name} not in adapters: {self.adapters.keys()}')
            return

        from .mapping import SWIFT_MAPPING
        SWIFT_MAPPING[self.adapters[adapter_name].config.swift_type][1]\
            .activate_adapter(self.base_model, adapter_name, False, offload=offload)
        self.active_adapters = self.active_adapters - {adapter_name}

    def get_trainable_parameters(self):
        """
        Get the content of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.base_model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, 'ds_numel'):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return f'trainable params: {trainable_params:,d} || all params: {all_param:,d} ' \
               f'|| trainable%: {100 * trainable_params / all_param:.4f}' \
               '|| cuda memory: ' \
               f'{sum([torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())])/1024/1024/1024:.2f}' \
               'GiB.'


class Swift:
    """The Wrapper to use both Peft and Swift tuners."""

    @staticmethod
    def prepare_model(model: Union[nn.Module, SwiftModel],
                      config: Union[SwiftConfig, PeftConfig,
                                    Dict[str, SwiftConfig]], **kwargs):
        """Prepare a model by the input config.

        Args:
            model(`Union[nn.Module, 'SwiftModel']`): The model to be tuned.
            config(`Union[SwiftConfig, PeftConfig, Dict[str, SwiftConfig]]`): The config or config dict, can be either
                SwiftConfigs or PeftConfigs
            **kwargs:
                Extra kwargs needed by SwiftModel or PeftModel.
        Returns:
            The model wrapped by SwiftModel or PeftModel.
        """

        if isinstance(config, (SwiftConfig, dict)):
            return SwiftModel(model, config, **kwargs)
        else:
            return get_peft_model(model, config, **kwargs)

    @staticmethod
    def merge_and_unload(model: Union[PeftModel, SwiftModel], **kwargs):
        """Merge tuners into the base model and unload them.

        Args:
            model(`Union[PeftModel, SwiftModel]`): The model instance with tuners
            kwargs:
                adapter_name(`Union[str, List[str]]`): The adapter_name to unload, only supported in swift tuners.

        """
        from peft import PeftModel as _PeftModel
        if isinstance(model, _PeftModel):
            model.merge_and_unload()
        elif isinstance(model, SwiftModel):
            from swift import LoRAConfig
            from swift.tuners import LoRA
            adapter_name = kwargs.get('adapter_name', None)
            if isinstance(adapter_name, str):
                adapter_name = [adapter_name]
            for adapter, output in model.adapters.items():
                if isinstance(output.config,
                              LoRAConfig) and (adapter_name is None
                                               or adapter in adapter_name):
                    LoRA.unpatch_lora(model, output.config, adapter)

    @staticmethod
    def merge(model: Union[PeftModel, SwiftModel], **kwargs):
        """Merge tuners into the base model, will not unload them.

        Args:
            model(`Union[PeftModel, SwiftModel]`): The model instance with tuners
        """
        from .lora_layers import LoraLayer, LoRALayer
        for sub_module in model.modules():
            if isinstance(sub_module, (LoraLayer, LoRALayer)):
                sub_module.merge(**kwargs)

    @staticmethod
    def unmerge(model: Union[PeftModel, SwiftModel], **kwargs):
        """Unmerge tuners from the base model

        Args:
            model(`Union[PeftModel, SwiftModel]`): The model instance with tuners
        """
        from .lora_layers import LoraLayer, LoRALayer
        for sub_module in model.modules():
            if isinstance(sub_module, (LoraLayer, LoRALayer)):
                sub_module.unmerge(**kwargs)

    @staticmethod
    def save_to_peft_format(ckpt_dir: str, output_dir: str) -> None:
        """Save swift format to peft format

        Args:
            ckpt_dir(`str`): Original swift output dir
            output_dir(`str`): Converted peft format dir
        """
        assert ckpt_dir and output_dir, 'Please pass in valid ckpt_dir and output_dir.'
        assert os.path.exists(
            ckpt_dir), f'ckpt_dir: {ckpt_dir} must exists in local disk.'
        if os.path.exists(os.path.join(ckpt_dir, SwiftModel.EXTRA_STATE_DIR)):
            raise AssertionError(
                'Cannot transfer to peft format, because you are additional state dicts.'
            )

        adapter_names = [
            sub_dir for sub_dir in os.listdir(ckpt_dir)
            if os.path.isfile(os.path.join(ckpt_dir, sub_dir, CONFIG_NAME))
        ]

        def has_custom_content(_json):
            if _json.get('swift_type',
                         _json.get('peft_type')) != SwiftTuners.LORA:
                logger.warn('Only LoRA can be converted to peft format')
                return True

            from swift import LoRAConfig
            return not LoRAConfig(**_json).can_be_saved_to_peft()

        for adapter in adapter_names:
            with open(os.path.join(ckpt_dir, adapter, CONFIG_NAME)) as f:
                _json = json.load(f)
                if has_custom_content(_json):
                    raise AssertionError(
                        'Cannot transfer to peft format, '
                        'because you have special parameters or adapter types.'
                    )

        os.makedirs(output_dir, exist_ok=True)
        if ckpt_dir != output_dir:
            shutil.copytree(ckpt_dir, output_dir, dirs_exist_ok=True)

        for adapter in adapter_names:
            safe_serialization = os.path.isfile(
                os.path.join(output_dir, adapter, SAFETENSORS_WEIGHTS_NAME))
            state_dict = SwiftModel.load_state_file(
                os.path.join(output_dir, adapter))
            new_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('base_model.model.'):
                    key = 'base_model.model.' + key
                key = key.replace(f'lora_A.{adapter}.', 'lora_A.')
                key = key.replace(f'lora_B.{adapter}.', 'lora_B.')
                key = key.replace(f'lora_embedding_A.{adapter}.',
                                  'lora_embedding_A.')
                key = key.replace(f'lora_embedding_B.{adapter}.',
                                  'lora_embedding_B.')
                key = key.replace(f'lora_magnitude_vector.{adapter}',
                                  'lora_magnitude_vector')
                new_state_dict[key] = value
            state_dict = new_state_dict
            SwiftModel._save_state_dict(state_dict,
                                        os.path.join(output_dir, adapter),
                                        safe_serialization)
            from swift import LoRAConfig
            with open(os.path.join(output_dir, adapter, CONFIG_NAME)) as f:
                _json = json.load(f)
                peft_config = LoRAConfig(**_json).to_peft_config()
            peft_config.save_pretrained(os.path.join(output_dir, adapter))

        if 'default' in adapter_names:
            shutil.move(
                os.path.join(output_dir, 'default', CONFIG_NAME),
                os.path.join(output_dir, CONFIG_NAME))
            state_dict = SwiftModel.load_state_file(
                os.path.join(output_dir, 'default'))
            safe_serialization = os.path.isfile(
                os.path.join(output_dir, 'default', SAFETENSORS_WEIGHTS_NAME))
            SwiftModel._save_state_dict(state_dict, output_dir,
                                        safe_serialization)
            shutil.rmtree(os.path.join(output_dir, 'default'))

    @staticmethod
    def from_pretrained(model: Union[nn.Module, SwiftModel],
                        model_id: str = None,
                        adapter_name: Union[str, List[str], Dict[str,
                                                                 str]] = None,
                        revision: str = None,
                        **kwargs):
        """Prepare a model by a model_id in the ModelScope hub or a local dir.

        Args:
            model(`Union[nn.Module, 'SwiftModel']`): The model to be tuned.
            model_id(`str`): The model id of the modelhub or a local dir containing the configs/weights.
            adapter_name(`str`, `optional`): The adapter_name to use.
            revision(`str`, `optional`): The model revision if the model_id is a model id of the modelhub.
            **kwargs:
                Extra kwargs needed by ``SwiftModel.from_pretrained`` or ``PeftModel.from_pretrained``.
        Returns:
            The model wrapped by SwiftModel or PeftModel.
        """
        if not os.path.exists(model_id):
            model_id = snapshot_download(model_id, revision=revision)
        is_peft_model = False
        if os.path.exists(os.path.join(model_id, CONFIG_NAME)):
            with open(os.path.join(model_id, CONFIG_NAME), 'r') as f:
                _json = json.load(f)
            is_peft_model = SWIFT_TYPE_KEY not in _json

        _name = adapter_name if isinstance(
            adapter_name, str) or adapter_name is None else adapter_name[0] \
            if isinstance(adapter_name, list) else list(adapter_name.keys())[0]
        _name = _name or ''
        if os.path.exists(os.path.join(model_id, _name, CONFIG_NAME)):
            with open(os.path.join(model_id, _name, CONFIG_NAME), 'r') as f:
                _json = json.load(f)
            is_peft_model = SWIFT_TYPE_KEY not in _json and 'extra_state_keys' not in _json
        if is_peft_model:
            return PeftModel.from_pretrained(
                model,
                model_id,
                revision=revision,
                adapter_name=adapter_name or 'default',
                **kwargs)
        else:
            return SwiftModel.from_pretrained(
                model,
                model_id,
                revision=revision,
                adapter_name=adapter_name,
                **kwargs)
