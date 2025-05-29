# LLaMA-Factory Training Example

## Environment Setup

### Install dependencies:

```bash
git clone http://github.com/7Sageer/llama-factory-example
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
pip install -r requirements.txt
```

### Prepare dataset:

Place your dataset files in the `data` directory, or specify the dataset path using environment variables.

## Configuration

All configurations can be set through environment variables. Here's a summary of available parameters:

### Parameter Tables

#### Path Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `ROOT_DIR` | Project root directory | Current directory |
| `LLAMA_FACTORY_DIR` | LLaMA-Factory directory | `${ROOT_DIR}/../LLaMA-Factory` |
| `MODEL_PATH` | Pre-trained model path | `${ROOT_DIR}/models/llm4decompile-1.3b-v1.5` |
| `DATASET_DIR` | Dataset directory. This directory should contain your `dataset_info.json` and the actual dataset files. | `${ROOT_DIR}/data` |
| `OUTPUT_DIR` | Output model directory | `${ROOT_DIR}/output_models/${exp_id}` |
| `WANDB_PROJECT` | WandB project name | `LLM4Binary` |
| `WANDB_MODE` | WandB mode | `online` |
| `WANDB_API_KEY` | WandB API key | `your_api_key_here` **Replace with your actual API key** |
| `WANDB_BASE_URL` | WandB base URL | `https://api.wandb.ai` |
| `WANDB_DISABLED` | Disable WandB | `false` (enabled) |
| `DEEPSPEED_PORT` | DeepSpeed port | `11000` |
| `HOSTFILE` | Host file path for distributed training | Not set |
| `EXP_ID` | Experiment ID | `deepseek-1.3b-llm4decompile-v15-llm4binary-v2` |
| `CUTOFF_LEN` | Sequence truncation length | `4096` |
| `MAX_GRAD_NORM` | Gradient clipping value | `1.0` |
| `NUM_WORKERS` | Preprocessing worker threads | `256` |
| `BATCH_SIZE` | Per-device training batch size | `16` |
| `GRAD_ACCUM_STEPS` | Gradient accumulation steps | `16` |
| `LEARNING_RATE` | Learning rate | `5e-6` |
| `LR_SCHEDULER` | Learning rate scheduler type | `cosine` |
| `LOGGING_STEPS` | Logging frequency (steps) | `1` |
| `WARMUP_RATIO` | Warmup ratio | `0.025` |
| `SAVE_STEPS` | Model saving frequency (steps) | `20` |
| `SAVE_TOTAL_LIMIT` | Maximum number of saved models | `10` |
| `FLASH_ATTN` | Flash Attention type | `fa2` |
| `MAX_SAMPLES` | Maximum number of samples | `20000000` |
| `NUM_EPOCHS` | Number of training epochs | `1.0` |
| `BF16` | Enable BF16 precision | Not set (disabled) |

## Running Training

```bash
# Run with default configuration
bash run_training.sh

# Or customize configuration
export MODEL_PATH="/path/to/your/model"
export DATASET_DIR="/path/to/your/data_directory"
export BATCH_SIZE=8
export NUM_EPOCHS=3
bash run_training.sh
```

## Example Data

The `data/llm4binary_v1_example.json` file contains example data.
