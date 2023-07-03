#!/usr/bin/bash

LLAMA_PATH="$1"
PRETRAINED_PATH="$2" # path to pre-trained checkpoint
CONFIG="$3"
OUTPUT_DIR="$4"

mkdir -p "$OUTPUT_DIR"

python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
 main_finetune.py --data_config "$CONFIG" --batch_size 1 --accum_iter 32 \
 --epochs 20 --warmup_epochs 1 --blr 10e-4 --weight_decay 0.05 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR" \
 --pretrained_path "$PRETRAINED_PATH"
