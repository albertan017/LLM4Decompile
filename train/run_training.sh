#!/bin/bash

# Set default values that can be overridden by environment variables
DEEPSPEED_PORT=${DEEPSPEED_PORT:-11000}
deepspeed_args="--master_port=${DEEPSPEED_PORT}" # Default argument
# If you need to use a hostfile, set the HOSTFILE environment variable
# Example: export HOSTFILE=/path/to/hostfile
# The script will automatically add the hostfile parameter
if [ ! -z "${HOSTFILE}" ]; then
  deepspeed_args="${deepspeed_args} --hostfile=${HOSTFILE}"
fi

# Set Weights & Biases (WandB) environment variables
export WANDB_PROJECT=${WANDB_PROJECT:-"LLM4Binary"}
export WANDB_MODE=${WANDB_MODE:-"online"}
export WANDB_API_KEY=${WANDB_API_KEY:-"your_api_key_here"}
export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}
# WandB is enabled by default, set WANDB_DISABLED=true to disable it
export WANDB_DISABLED=${WANDB_DISABLED:-"false"}

# Set path variables that can be overridden by environment variables
ROOT_DIR=${ROOT_DIR:-$(pwd)}
LLAMA_FACTORY_DIR=${LLAMA_FACTORY_DIR:-"${ROOT_DIR}/../LLaMA-Factory"}

# Model and dataset configuration
model_name_or_path=${MODEL_PATH:-"${ROOT_DIR}/models/llm4decompile-1.3b-v1.5"}
dataset="llm4binary_v1"
exp_id=${EXP_ID:-"deepseek-1.3b-llm4decompile-v15-llm4binary-v2"}
dataset_dir=${DATASET_DIR:-"${ROOT_DIR}/data"}
output_dir=${OUTPUT_DIR:-"${ROOT_DIR}/output_models/${exp_id}"}
mkdir -p "${output_dir}"

# --- Training Command Structure ---
deepspeed ${deepspeed_args} \
    ${LLAMA_FACTORY_DIR}/src/train.py \
    --deepspeed ${LLAMA_FACTORY_DIR}/examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path ${model_name_or_path} \
    --dataset ${dataset} \
    --dataset_dir ${dataset_dir} \
    --template empty \
    --finetuning_type full \
    --output_dir ${output_dir} \
    --gradient_checkpointing 1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len ${CUTOFF_LEN:-4096} \
    --max_grad_norm ${MAX_GRAD_NORM:-1.0} \
    --preprocessing_num_workers ${NUM_WORKERS:-256} \
    --per_device_train_batch_size ${BATCH_SIZE:-16} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS:-16} \
    --learning_rate ${LEARNING_RATE:-5e-6} \
    --lr_scheduler_type ${LR_SCHEDULER:-"cosine"} \
    --logging_steps ${LOGGING_STEPS:-1} \
    --warmup_ratio ${WARMUP_RATIO:-0.025} \
    --run_name ${exp_id} \
    --save_steps ${SAVE_STEPS:-20} \
    --save_total_limit ${SAVE_TOTAL_LIMIT:-10} \
    --flash_attn ${FLASH_ATTN:-fa2} \
    --max_samples ${MAX_SAMPLES:-20000000} \
    --num_train_epochs ${NUM_EPOCHS:-1.0} \
    --plot_loss \
    ${BF16:+--bf16} \
| tee ${output_dir}/train.log 2>${output_dir}/train.err
# --- End of Training Command Structure ---

# Notes:
# 1. All parameters can be overridden by environment variables
# 2. For boolean parameters like bf16, you can control them by setting or not setting environment variables
#    Example: export BF16=1 to enable bf16, or leave it unset to disable
