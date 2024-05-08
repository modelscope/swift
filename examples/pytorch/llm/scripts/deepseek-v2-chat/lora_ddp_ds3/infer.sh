# Experimental environment: 8*A100

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
    --ckpt_dir output/deepseek-v2-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
    --use_flash_attn true \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --device_map_path ./deepseek2_device_map.json
