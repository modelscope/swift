# Experimental environment: A10
# 18GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_sft.py \
    --model_id_or_path Shanghai_AI_Laboratory/internlm-chat-20b \
    --model_revision master \
    --sft_type lora \
    --tuner_backend swift \
    --template_type internlm \
    --dtype bf16 \
    --output_dir output \
    --dataset sql-create-context-en \
    --train_dataset_sample 20000 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --quantization_bit 4 \
    --bnb_4bit_comp_dtype bf16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules AUTO \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --push_to_hub false \
    --hub_model_id internlm-20b-chat-qlora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
