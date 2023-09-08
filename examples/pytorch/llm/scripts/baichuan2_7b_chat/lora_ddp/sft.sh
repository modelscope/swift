# Experimental environment: 2 * A100
nproc_per_node=2
CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    src/llm_sft.py \
    --model_type baichuan2-7b-chat \
    --sft_type lora \
    --template_type baichuan \
    --dtype bf16 \
    --output_dir runs \
    --ddp_backend nccl \
    --dataset alpaca-en,alpaca-zh \
    --dataset_sample 20000 \
    --num_train_epochs 1 \
    --max_length 1024 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules W_pack o_proj \
    --gradient_checkpointing false \
    --batch_size 1 \
    --weight_decay 0. \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --push_to_hub false \
    --hub_model_id baichuan2-7b-chat-lora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
