# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import importlib
import os
import re
import shutil
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import safetensors
import torch
import transformers
from datasets import Dataset as HfDataset
from packaging import version
from peft import PeftModel
from torch import nn
from torch.nn import Module
from transformers import (PreTrainedModel, PreTrainedTokenizerBase,
                          is_bitsandbytes_available)
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import unwrap_model
from transformers.trainer import ADAPTER_CONFIG_NAME  # noqa
from transformers.trainer import (ADAPTER_SAFE_WEIGHTS_NAME,
                                  ADAPTER_WEIGHTS_NAME, CONFIG_NAME,
                                  PREFIX_CHECKPOINT_DIR, SAFE_WEIGHTS_NAME,
                                  TRAINER_STATE_NAME, TRAINING_ARGS_NAME,
                                  WEIGHTS_NAME, IntervalStrategy, Trainer,
                                  TrainerCallback, is_peft_available)
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from swift.hub import Repository
from swift.hub.check_model import check_local_model_is_latest
from swift.tuners import SwiftModel
from swift.utils import check_json_format, create_ms_repo, get_logger
from swift.utils.constants import Invoke
from .optimizers.galore import create_optimizer_and_scheduler
from .utils import (can_return_loss, find_labels, get_function,
                    is_instance_of_ms_model)

logger = get_logger()


def _push_to_hub(self: Repository,
                 commit_message: str = 'Commit files to Modelscope Hub',
                 **kwargs):
    blocking = kwargs.get('blocking', True)
    self.push(commit_message)
    if not blocking:
        # Compatible with transformers
        return None, None
    else:
        return None


class PushToMsHubMixin:
    repo: Repository

    def _add_patterns_to_file(self,
                              file_name: str,
                              patterns: List[str],
                              commit_message: Optional[str] = None) -> None:
        # Make sure we only do this on the main process
        if not self.is_world_process_zero():
            return
        if isinstance(patterns, str):
            patterns = [patterns]
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to {file_name}'

        # Get current file content
        repo_dir = self.repo.model_dir
        file_path = os.path.join(repo_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
        else:
            current_content = ''
        # Add the patterns to file
        content = current_content
        for pattern in patterns:
            if pattern not in content:
                if len(content) > 0 and not content.endswith('\n'):
                    content += '\n'
                content += f'{pattern}\n'

        # Write the file if it has changed
        if content != current_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                logger.debug(f'Writing {file_name} file. Content: {content}')
                f.write(content)
        self.repo.push(commit_message)

    def _add_patterns_to_gitignore(
            self,
            patterns: List[str],
            commit_message: Optional[str] = None) -> None:
        self._add_patterns_to_file('.gitignore', patterns, commit_message)

    def _add_patterns_to_gitattributes(
            self,
            patterns: List[str],
            commit_message: Optional[str] = None) -> None:
        new_patterns = []
        suffix = 'filter=lfs diff=lfs merge=lfs -text'
        for pattern in patterns:
            if suffix not in pattern:
                pattern = f'{pattern} {suffix}'
            new_patterns.append(pattern)
        file_name = '.gitattributes'
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to {file_name}'
        self._add_patterns_to_file(file_name, new_patterns, commit_message)

    def init_hf_repo(self) -> None:
        """init ms repo. Compatible with transformers>=4.34"""
        self.init_git_repo(at_init=True)

    def init_git_repo(self, at_init: bool = False) -> None:
        if not self.is_world_process_zero():
            return
        if (os.path.exists(self.args.output_dir)
                and os.listdir(self.args.output_dir)
                and self.args.overwrite_output_dir and at_init):
            # directory not empty.
            shutil.rmtree(self.args.output_dir)
        self.args.hub_model_id = create_ms_repo(self.args.hub_model_id,
                                                self.args.hub_token,
                                                self.args.hub_private_repo)
        self.repo = Repository(self.args.output_dir, self.args.hub_model_id)
        self._add_patterns_to_gitattributes(['*.safetensors', '*.bin', '*.pt'])
        self.repo.push_to_hub = MethodType(_push_to_hub, self.repo)
        self.repo.local_dir = self.repo.model_dir  # hf compatibility

        # By default, ignore the checkpoint folders
        if self.args.push_hub_strategy != 'all_checkpoints':
            self._add_patterns_to_gitignore(
                ['checkpoint-*/', 'tmp-checkpoint-*/'])

        # Add 'runs/' to .gitignore, ignore tensorboard files
        self._add_patterns_to_gitignore(['runs/'])

        # Add '*.sagemaker' to .gitignore if using SageMaker
        if os.environ.get('SM_TRAINING_ENV'):
            self._add_patterns_to_gitignore(
                ['*.sagemaker-uploading', '*.sagemaker-uploaded'],
                'Add `*.sagemaker` patterns to .gitignore')

        self.push_in_progress = None

    def push_to_hub(self,
                    commit_message: str = 'End of training',
                    **kwargs) -> None:
        # user calls manually `push_to_hub` with `self.args.push_to_hub = False`
        create_model_card = kwargs.pop('create_model_card', None)
        if not hasattr(self, 'repo'):
            self.init_git_repo()
        self.save_model(_internal_call=True)

        if not self.is_world_process_zero():
            return

        self.repo.push_to_hub(commit_message, **kwargs)
        # push separately the model card to be independant from the rest of the model
        readme_path = os.path.join(self.args.output_dir, 'README.md')
        if create_model_card is None:
            create_model_card = not os.path.exists(readme_path)
        if create_model_card and self.args.should_save:
            model_name = kwargs.pop('model_name', None)
            if model_name is None and self.args.should_save:
                if self.args.hub_model_id is not None:
                    model_name = self.args.hub_model_id.split('/')[-1]
                else:
                    model_name = os.path.basename(self.args.output_dir)
            self.create_model_card(model_name=model_name, **kwargs)
            self.repo.push_to_hub('update model card README.md', **kwargs)

    def _push_from_checkpoint(self, checkpoint_folder: str) -> None:
        """Compatible with transformers>=4.32"""
        # Only push from one node.
        if not self.is_world_process_zero(
        ) or self.args.push_hub_strategy == 'end':
            return
        output_dir = self.args.output_dir
        # To avoid a new synchronization of all model weights, we just copy the file from the checkpoint folder
        modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
        if is_peft_available():
            modeling_files.extend([
                ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME,
                ADAPTER_SAFE_WEIGHTS_NAME
            ])
        for modeling_file in modeling_files:
            if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
                shutil.copy(
                    os.path.join(checkpoint_folder, modeling_file),
                    os.path.join(output_dir, modeling_file))
        # Saving the tokenizer is fast and we don't know how many files it may have spawned, so we resave it to be sure.
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # Same for the training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        try:
            if self.args.push_hub_strategy == 'checkpoint':
                # Temporarily move the checkpoint just saved for the push
                tmp_checkpoint = os.path.join(output_dir, 'last-checkpoint')
                # We have to remove the "last-checkpoint" dir if it exists, otherwise the checkpoint is moved as a
                # subfolder.
                if os.path.isdir(tmp_checkpoint):
                    shutil.rmtree(tmp_checkpoint)
                shutil.move(checkpoint_folder, tmp_checkpoint)

            if self.args.save_strategy == IntervalStrategy.STEPS:
                commit_message = f'Training in progress, step {self.state.global_step}'
            else:
                commit_message = f'Training in progress, epoch {int(self.state.epoch)}'
            if self.args.push_hub_strategy == 'push_best':
                folder, checkpoint_name = os.path.split(checkpoint_folder)
                checkpoint_name = checkpoint_name.replace(
                    'tmp-checkpoint-', 'checkpoint-')
                last_model_checkpoint = os.path.join(folder, checkpoint_name)
                if last_model_checkpoint == self.state.best_model_checkpoint:
                    self.repo.push_to_hub(
                        commit_message=commit_message,
                        blocking=False,
                        auto_lfs_prune=True)
            else:
                self.repo.push_to_hub(
                    commit_message=commit_message,
                    blocking=False,
                    auto_lfs_prune=True)
        except Exception as e:
            logger.error(f'Error when pushing to hub: {e}')
        finally:
            if self.args.push_hub_strategy == 'checkpoint':
                # Move back the checkpoint to its place
                shutil.move(tmp_checkpoint, checkpoint_folder)


class SwiftMixin:

    def __init__(self,
                 model: Union[PreTrainedModel, Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[HfDataset] = None,
                 eval_dataset: Optional[Union[HfDataset,
                                              Dict[str, HfDataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction],
                                                    Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer,
                                   torch.optim.lr_scheduler.LambdaLR] = (None,
                                                                         None),
                 preprocess_logits_for_metrics: Optional[Callable[
                     [torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 **kwargs) -> None:
        check_model = kwargs.pop('check_model', True)
        if check_model and hasattr(model, 'model_dir'):
            check_local_model_is_latest(
                model.model_dir,
                user_agent={
                    Invoke.KEY:
                    Invoke.LOCAL_TRAINER,
                    Invoke.THIRD_PARTY:
                    kwargs.pop(Invoke.THIRD_PARTY, Invoke.SWIFT),
                })

        # Compatible with transformers>=4.34
        from swift.tuners import SwiftModel
        is_quantized = getattr(model, 'is_quantized', False)
        _hf_peft_config_loaded = getattr(model, '_hf_peft_config_loaded',
                                         False)
        use_swift = isinstance(model, SwiftModel)
        if is_quantized and use_swift:
            model._hf_peft_config_loaded = True
        # mro
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs)
        if is_quantized and use_swift:
            model._hf_peft_config_loaded = _hf_peft_config_loaded

        if get_function(model.__class__.forward) is not get_function(
                model.forward):
            self.label_names = find_labels(model)
            self.can_return_loss = can_return_loss(model)

    @staticmethod
    def _create_configuration_file(model: Module, output_dir: str) -> None:
        cfg = getattr(model, 'cfg', {})
        configuration_path = os.path.join(output_dir, 'configuration.json')
        new_cfg = {}
        if os.path.exists(configuration_path):
            with open(configuration_path, 'r', encoding='utf-8') as f:
                new_cfg = json.load(f)

        if 'framework' not in new_cfg:
            new_cfg['framework'] = cfg.get('framework', 'pytorch')
        if 'task' not in new_cfg:
            new_cfg['task'] = cfg.get('task', 'text-generation')
        with open(configuration_path, 'w', encoding='utf-8') as f:
            json.dump(new_cfg, f, ensure_ascii=False, indent=4)

    def _add_adapter_cfg(self, output_dir: str) -> None:
        if not hasattr(self, 'sft_args'):
            return
        sft_args = self.sft_args
        if sft_args.sft_type == 'full':
            return
        configuration_path = os.path.join(output_dir, 'configuration.json')
        new_cfg = {}
        if os.path.exists(configuration_path):
            with open(configuration_path, 'r', encoding='utf-8') as f:
                new_cfg = json.load(f)

        need_to_save = [
            'model_id_or_path', 'model_revision', 'sft_type', 'tuner_backend',
            'template_type', 'dtype', 'system'
        ]
        quantization_bit = sft_args.quantization_bit
        if quantization_bit > 0:
            need_to_save += [
                'quantization_bit', 'bnb_4bit_comp_dtype',
                'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant'
            ]
        adapter_cfg = {}
        for k in need_to_save:
            adapter_cfg[k] = getattr(sft_args, k)
        new_cfg['adapter_cfg'] = adapter_cfg
        with open(configuration_path, 'w', encoding='utf-8') as f:
            json.dump(new_cfg, f, ensure_ascii=False, indent=4)

    def _save_sft_args(self, output_dir: str) -> None:
        sft_args = getattr(self, 'sft_args', None)
        if sft_args is None:
            return
        fpath = os.path.join(output_dir, 'sft_args.json')
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(
                check_json_format(self.sft_args.__dict__),
                f,
                ensure_ascii=False,
                indent=2)
        return

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Compatible with swift and peft"""
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # configuration.json
        model_dir = getattr(self.model, 'model_dir', None)
        if model_dir is not None:
            src_path = os.path.join(model_dir, 'configuration.json')
            dst_path = os.path.join(output_dir, 'configuration.json')
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
        else:
            self._create_configuration_file(self.model, output_dir)
        self._add_adapter_cfg(output_dir)
        self._save_sft_args(output_dir)
        # generation_config
        generation_config = getattr(self.args, 'generation_config', None)
        if generation_config is not None:
            generation_config.save_pretrained(output_dir)
        # model
        supported_classes = (SwiftModel, PreTrainedModel, PeftModel)
        save_safetensors = self.args.save_safetensors
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            _unwrap_model = unwrap_model(self.model)
            if isinstance(_unwrap_model, supported_classes):
                _unwrap_model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=save_safetensors)
            else:
                logger.info(
                    'Trainer.model is not a `PreTrainedModel`, only saving its state dict.'
                )
                if save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, 'model.safetensors'))
                else:
                    torch.save(state_dict,
                               os.path.join(output_dir, 'pytorch_model.bin'))
        elif is_instance_of_ms_model(self.model):
            PreTrainedModel.save_pretrained(
                self.model,
                output_dir,
                state_dict=state_dict,
                safe_serialization=save_safetensors)
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=save_safetensors)
        # tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # training_args.bin
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        # additional files
        additional_files = getattr(self.args, 'additional_saved_files', [])
        if model_dir is not None:
            for file in additional_files:
                src_path = os.path.join(model_dir, file)
                dst_path = os.path.join(output_dir, file)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)

    def _save_checkpoint(self, model, trial, metrics=None):
        self.state.last_model_checkpoint = os.path.join(
            self.args.output_dir, f'checkpoint-{self.state.global_step}')
        logger.info(
            f'Saving model checkpoint to {self.state.last_model_checkpoint}')
        if version.parse(transformers.__version__) >= version.parse(
                '4.36') or not self.args.save_only_model:
            return super()._save_checkpoint(model, trial, metrics)
        else:
            return self._save_only_model(model, trial, metrics)

    def _save_only_model(self, model, trial, metrics=None):
        # Save model checkpoint
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith('eval_'):
                metric_to_check = f'eval_{metric_to_check}'
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(
                os.path.join(output_dir, TRAINER_STATE_NAME))

        # push to hub
        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        train_sampler_random = self.args.train_sampler_random
        if train_sampler_random:
            return super()._get_train_sampler()
        else:
            return self._get_eval_sampler(self.train_dataset)

    def _load_from_checkpoint(self,
                              resume_from_checkpoint: str,
                              model=None) -> None:
        if model is None:
            model = self.model
        if not isinstance(model, SwiftModel):
            # Avoid throwing exceptions
            return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _sorted_checkpoints(self,
                            output_dir=None,
                            checkpoint_prefix=PREFIX_CHECKPOINT_DIR,
                            use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [
            str(x) for x in Path(output_dir).glob(f'{checkpoint_prefix}-*')
            if os.path.isdir(x)
        ]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append(
                    (os.path.getmtime(path), path))
            else:
                regex_match = re.match(f'.*{checkpoint_prefix}-([0-9]+)', path)
                if regex_match is not None and regex_match.groups(
                ) is not None:
                    ordering_and_checkpoint_path.append(
                        (int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [
            checkpoint[1] for checkpoint in checkpoints_sorted
        ]
        # Make sure we don't delete the best model.
        if (self.state.best_model_checkpoint is not None and str(
                Path(self.state.best_model_checkpoint)) in checkpoints_sorted):
            best_model_index = checkpoints_sorted.index(
                str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[
                    i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _load_best_model(self):
        # Compatible with transformers>=4.35 (deepspeed)
        try:
            model = self.model
            if isinstance(model, SwiftModel):
                logger.info(
                    f'Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).'
                )
                adapters = model.adapters
                for adapter_name in adapters.keys():
                    sub_folder = os.path.join(self.state.best_model_checkpoint,
                                              adapter_name)
                    state_dict = SwiftModel.load_state_file(
                        sub_folder, device='cpu')
                    if state_dict is not None:
                        self.model.load_state_dict(
                            state_dict,
                            strict=False,
                            adapter_name=adapter_name)
                state_dict = SwiftModel.load_state_file(
                    self.state.best_model_checkpoint, device='cpu')
                if state_dict is not None:
                    self.model.load_state_dict(
                        state_dict, strict=False, adapter_name='default')
            else:
                super()._load_best_model()
        except ValueError as e:
            logger.warning(e)

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        if self.control.should_log:
            self.control.should_log = False
            logs: Dict[str, float] = {}
            metrics_log = {'loss': tr_loss}  # loss first
            if hasattr(self, '_custom_metrics'):
                metrics_log.update(self._custom_metrics)
                self._custom_metrics = {}
            for k, v in metrics_log.items():
                # all_gather + mean() to get average loss over all processes
                v_scalar = self._nested_gather(v).mean().item()
                if k == 'loss':
                    self._total_loss_scalar += v_scalar
                logs[k] = round(
                    v_scalar /
                    (self.state.global_step - self._globalstep_last_logged), 8)
            if version.parse(
                    transformers.__version__) >= version.parse('4.38'):
                grad_norm = args[0]
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()
                if grad_norm is not None:
                    logs['grad_norm'] = grad_norm
            logs['learning_rate'] = self._get_learning_rate()

            tr_loss -= tr_loss
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)
        super()._maybe_log_save_evaluate(tr_loss, *args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if hasattr(self.args, 'galore_config'):
            optimizer, lr_scheduler = create_optimizer_and_scheduler(
                self.model,
                self.args,
                self.args.galore_config,
                num_training_steps,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            self.create_optimizer()
            self.create_scheduler(
                num_training_steps=num_training_steps,
                optimizer=self.optimizer)

    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            if version.parse(
                    transformers.__version__) < version.parse('4.34.0'):
                logger.warning(
                    f'If you are using lora+, please remember using transformers>=4.34.0, '
                    f'but now is {transformers.__version__}')
                return super().create_optimizer()

            optimizer_grouped_parameters = None
            if hasattr(self.model, 'create_optimizer_param_groups'):
                # Lora+ parameter groups
                optimizer_grouped_parameters = self.model.create_optimizer_param_groups(
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay)

            if optimizer_grouped_parameters is None:
                # Default parameter groups
                decay_parameters = self.get_decay_parameter_names(opt_model)
                optimizer_grouped_parameters = [
                    {
                        'params': [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        'weight_decay':
                        self.args.weight_decay,
                    },
                    {
                        'params': [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        'weight_decay':
                        0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                           **optimizer_kwargs)
        return self.optimizer
