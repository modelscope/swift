# Experimental environment: A10
# If you want to merge LoRA weight and save it, you need to set `--merge_lora_and_save true`.
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --model_id_or_path qwen/Qwen-7B \
    --model_revision master \
    --sft_type lora \
    --tuner_backend swift \
    --template_type default-generation \
    --dtype bf16 \
    --ckpt_dir "output/qwen-7b/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset dureader-robust-zh \
    --max_length 2048 \
    --use_flash_attn false \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --do_sample true \
    --merge_lora_and_save false \
