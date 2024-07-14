# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Optional

import torch

from swift.utils import get_logger, get_main, get_model_info, push_to_ms_hub, seed_everything, show_layers
from .infer import merge_lora, prepare_model_template, save_checkpoint
from .utils import ExportArguments, Template, get_dataset, swift_to_peft_format
from .utils.model import safe_snapshot_download

logger = get_logger()

_args: Optional[ExportArguments] = None
template: Optional[Template] = None


def _get_dataset(*args, **kwargs):
    global _args, template
    assert _args is not None
    assert template is not None
    data = _args.dataset
    n_samples = _args.quant_n_samples
    block_size = _args.quant_seqlen

    # only use train_dataset
    dataset = get_dataset(
        data,
        0,
        _args.dataset_seed,
        check_dataset_strategy=_args.check_dataset_strategy,
        model_name=_args.model_name,
        model_author=_args.model_author)[0]
    logger.info(f'quant_dataset: {dataset}')
    dataset = dataset.shuffle()

    samples = []
    n_run = 0
    for data in dataset:
        input_ids = template.encode(data)[0].get('input_ids')
        if input_ids is None or len(input_ids) == 0:
            continue
        sample = torch.tensor(input_ids)
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=0)  # shape: [X]
    n_split = cat_samples.shape[0] // block_size
    logger.info(f'Split into {n_split} blocks')
    if _args.quant_method == 'awq':
        return [cat_samples[None, i * block_size:(i + 1) * block_size] for i in range(n_split)]
    else:  # gptq
        res = []
        for i in range(n_split):
            input_ids = cat_samples[None, i * block_size:(i + 1) * block_size]
            attention_mask = torch.ones_like(input_ids)
            res.append({'input_ids': input_ids, 'attention_mask': attention_mask})
        return res


def awq_model_quantize(awq_model, tokenizer) -> None:
    from awq.quantize import quantizer
    from transformers import AwqConfig
    assert _args is not None
    logger.info(f'Quantization dataset: {_args.dataset}')
    _origin_get_calib_dataset = quantizer.get_calib_dataset
    quantizer.get_calib_dataset = _get_dataset
    group_size = 128
    quant_config = {'zero_point': True, 'q_group_size': group_size, 'w_bit': _args.quant_bits, 'version': 'GEMM'}
    logger.info('Start quantizing the model...')
    awq_model.quantize(tokenizer, quant_config=quant_config)
    quantizer.get_calib_dataset = _origin_get_calib_dataset  # recover
    awq_model.model.config.quantization_config = AwqConfig(
        bits=_args.quant_bits, group_size=group_size, zero_point=True, version='GEMM')


def gptq_model_quantize(model, tokenizer):
    from optimum.gptq import GPTQQuantizer, quantizer
    global _args
    logger.info(f'Quantization dataset: {_args.dataset}')
    gptq_quantizer = GPTQQuantizer(bits=_args.quant_bits, dataset=','.join(_args.dataset))
    _origin_get_dataset = quantizer.get_dataset
    quantizer.get_dataset = _get_dataset
    logger.info('Start quantizing the model...')
    logger.warning('The process of packing the model takes a long time and there is no progress bar. '
                   'Please be patient and wait...')
    gptq_quantizer.quantize_model(model, tokenizer)
    quantizer.get_dataset = _origin_get_dataset  # recover
    return gptq_quantizer


def llm_export(args: ExportArguments) -> None:
    global _args, template
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    if args.to_peft_format:
        assert args.sft_type == 'lora', f'args.sft_type: {args.sft_type}'
        args.ckpt_dir = swift_to_peft_format(args.ckpt_dir)
    model_path = None
    if args.merge_lora:
        # fix parameter conflict
        quant_method = args.quant_method
        args.quant_method = None
        model_path = merge_lora(args, device_map=args.merge_device_map)
        args.quant_method = quant_method
    if args.to_ollama:
        if not model_path:
            model_path = safe_snapshot_download(args.model_type, args.model_id_or_path, revision=args.model_revision)
        logger.info(f'Exporting to ollama:')
        logger.info(f'If you have a gguf file, try to pass the file by :--gguf_file /xxx/xxx.gguf, '
                    f'else SWIFT will use the original(merged) model dir')
        os.makedirs(args.ollama_output_dir, exist_ok=True)
        model, template = prepare_model_template(args)
        with open(os.path.join(args.ollama_output_dir, 'Modelfile'), 'w') as f:
            f.write(f'FROM {model_path}\n')
            f.write(f'TEMPLATE """{{{{ if .System }}}}'
                    f'{template.system_prefix[0].replace("{{SYSTEM}}", "{{ .System }}")}'
                    f'{{{{ end }}}}')
            f.write(f'{{{{ if .Prompt }}}}'
                    f'{template.prompt[0].replace("{{QUERY}}", "{{ .Prompt }}")}'
                    f'{{{{ end }}}}')
            f.write('{{ .Response }}')
            f.write(template.suffix[0] + '"""\n')
            f.write(f'PARAMETER stop "{template.suffix[0]}"\n')
            if args.stop_words:
                for stop_word in args.stop_words:
                    f.write(f'PARAMETER stop "{stop_word}"\n')
        logger.info('Save Modelfile done, you can start ollama by:')
        logger.info('> ollama serve')
        logger.info('In another terminal:')
        logger.info('> ollama create my-custom-model '
                    f'-f {os.path.join(args.ollama_output_dir, "Modelfile")}')
        logger.info('> ollama run my-custom-model')
    elif args.quant_bits > 0:
        assert args.quant_output_dir is not None
        _args = args
        # assert args.quantization_bit == 0, f'args.quantization_bit: {args.quantization_bit}'
        assert args.sft_type == 'full', 'you need to merge lora'
        if args.quant_method == 'awq':
            from awq import AutoAWQForCausalLM
            model, template = prepare_model_template(
                args, device_map=args.quant_device_map, verbose=False, automodel_class=AutoAWQForCausalLM)
            awq_model_quantize(model, template.tokenizer)
            model.save_quantized(args.quant_output_dir)
        elif args.quant_method == 'gptq':
            model, template = prepare_model_template(args, device_map=args.quant_device_map, verbose=False)
            gptq_quantizer = gptq_model_quantize(model, template.tokenizer)
            model.config.quantization_config.pop('dataset', None)
            gptq_quantizer.save(model, args.quant_output_dir)
        elif args.quant_method == 'bnb':
            import bitsandbytes as bnb
            args.quantization_bit = args.quant_bits
            model, template = prepare_model_template(args, device_map=args.quant_device_map, verbose=False)
            if args.quant_bits == 8:
                dtype = torch.int8
            elif args.quant_bits == 4:
                dtype = torch.int4
            else:
                raise ValueError(f'Unsupported bits for bnb: {args.quant_bits}')
            model = bnb.quantization.quantize_model(model, dtype=dtype)
            model.save_pretrained(args.quant_output_dir)
        else:
            raise ValueError(f'args.quant_method: {args.quant_method}')

        logger.info(get_model_info(model))
        show_layers(model)
        logger.info('Saving quantized weights...')
        model_cache_dir = model.model_dir
        save_checkpoint(
            None,
            template.tokenizer,
            model_cache_dir,
            args.ckpt_dir,
            args.quant_output_dir,
            sft_args_kwargs={
                'dtype': args.dtype,
                'quant_method': args.quant_method
            })
        logger.info(f'Successfully quantized the model and saved in {args.quant_output_dir}.')
        args.ckpt_dir = args.quant_output_dir

    if args.push_to_hub:
        ckpt_dir = args.ckpt_dir
        if ckpt_dir is None:
            ckpt_dir = args.model_id_or_path
        assert ckpt_dir is not None, 'You need to specify `ckpt_dir`.'
        push_to_ms_hub(ckpt_dir, args.hub_model_id, args.hub_token, args.hub_private_repo, args.commit_message)


export_main = get_main(ExportArguments, llm_export)
