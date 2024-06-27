#!/bin/bash

# NCCL IB environment variables
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export OMP_NUM_THREADS=8

PROJECT_NAME="llm4decompile-6.7b-v2"
PARENT_SAVE_DIR="./output_models/"
PARENT_TENSORBOARD_DIR="./tensorboard/"
PARENT_CONFIG_FILE="./configs/"
PRETRAINED_MODEL_PATH="path/deepseek-coder-6.7b-base"

mkdir -p $PARENT_SAVE_DIR $PARENT_TENSORBOARD_DIR $PARENT_CONFIG_FILE

declare -a dataset=(
    "path_to_llm4decompile_data/arrow/part-00000"
)

FULL_PROJECT_NAME="${PROJECT_NAME}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"

colossalai run --nproc_per_node 8 --hostfile hostfile --master_port 30013 train.py \
    --pretrained $PRETRAINED_MODEL_PATH \
    --dataset ${dataset[@]} \
    --plugin "zero2" \
    --save_interval 400 \
    --save_dir $SAVE_DIR \
    --tensorboard_dir $TENSORBOARD_DIR \
    --config_file $CONFIG_FILE \
    --num_epochs 2 \
    --micro_batch_size 8 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --use_grad_checkpoint \
    --padding_mode "longest" \
    --max_length 4096 \
    --use_flash_attn \
    --pad_token "eos"
