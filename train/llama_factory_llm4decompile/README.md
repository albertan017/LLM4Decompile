# LLaMA-Factory Training Example

This document provides a detailed guide on how to use LLaMA-Factory for model training.

## 1\. Environment Setup

First, ensure your environment is set up correctly.

```bash
# Clone the example repository
git clone http://github.com/7Sageer/llama-factory-example
cd llama-factory-example

# Clone LLaMA-Factory as a dependency
# We place it in the parent directory to keep the project structure clean
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git ../LLaMA-Factory

# Install the required packages
pip install -r requirements.txt
```

## 2\. Prepare Data and Models

  - **Dataset**: Place your dataset files (e.g., `.json` files) and `dataset_info.json` into the `data` directory.
  - **Pre-trained Model**: Download or place your pre-trained model files into the `models` directory.

The file `data/llm4binary_v1_example.json` serves as a sample data example.

## 3\. Run Training

Training is executed by directly calling the `train.py` script from `LLaMA-Factory` with the `deepspeed` command. You need to modify the parameters in the command below according to your requirements.

### Training Command Template

This is a complete and ready-to-use training command template. Please **directly modify** the parameter values within it, and then execute it in your terminal.

```bash
# Before executing, first create the model output directory manually
# For example: mkdir -p output_models/my-first-experiment

deepspeed --master_port=11000 \
    ../LLaMA-Factory/src/train.py \
    --deepspeed ../LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path models/llm4decompile-1.3b-v1.5 \
    --dataset llm4binary_v1 \
    --dataset_dir data \
    --template empty \
    --finetuning_type full \
    --output_dir output_models/my-first-experiment \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 256 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_ratio 0.025 \
    --run_name "my-first-experiment" \
    --save_total_limit 10 \
    --gradient_checkpointing \
    --flash_attn "fa2" \
    --num_train_epochs 1.0 \
    --plot_loss \
    # --bf16 \ # If your hardware supports BF16 and you wish to enable it, uncomment this line
    # --hostfile /path/to/your/hostfile \ # For multi-node training, uncomment this line and provide the correct hostfile path
| tee output_models/my-first-experiment/train.log 2>output_models/my-first-experiment/train.err
```

### Parameter Descriptions

You can modify most of the parameters in the command above. The table below provides detailed descriptions of common parameters for your reference.

| Argument (`--argument`)       | Description                                                 | Default in Template                     |
| :---------------------------- | :---------------------------------------------------------- | :-------------------------------------- |
| `model_name_or_path`          | Path to a local pre-trained model or a HuggingFace model ID | `models/llm4decompile-1.3b-v1.5`        |
| `dataset`                     | Name of the dataset to use, must be defined in `dataset_info.json` | `llm4binary_v1`                         |
| `dataset_dir`                 | Directory where the dataset is located                      | `data`                                  |
| `output_dir`                  | The output directory for the model                          | `output_models/my-first-experiment`     |
| `run_name`                    | Experiment name displayed in monitoring tools like WandB    | `my-first-experiment`                   |
| `cutoff_len`                  | Maximum sequence truncation length                          | `4096`                                  |
| `per_device_train_batch_size` | Training batch size per GPU                                 | `16`                                    |
| `gradient_accumulation_steps` | Number of gradient accumulation steps                       | `16`                                    |
| `learning_rate`               | The learning rate                                           | `5e-6`                                  |
| `num_train_epochs`            | Total number of training epochs                             | `1.0`                                   |
| `bf16`                        | **(Flag)** Enables BF16 mixed-precision training            | Commented out                           |
| `logging_steps`               | Log every N steps                                           | `10`                                    |
| `save_steps`                  | Save a model checkpoint every N steps                       | `100`                                   |
| `save_total_limit`            | Maximum number of model checkpoints to save                 | `10`                                    |
| `flash_attn`                  | Version of Flash Attention to use (e.g., `fa2`)             | `"fa2"`                                 |
| `hostfile`                    | **(Multi-node)** Path to the DeepSpeed hostfile             | Commented out                           |

### Execution Example

Suppose you want to run a new experiment with a smaller batch size and 3 training epochs.

**Step 1: Create the output directory**

```bash
mkdir -p output_models/deepseek-3-epochs
```

**Step 2: Modify and execute the command**

Copy the **Training Command Template** from above and modify the following lines:

  - `--per_device_train_batch_size 8`
  - `--num_train_epochs 3.0`
  - `--output_dir output_models/deepseek-3-epochs`
  - `--run_name "deepseek-3-epochs"`
  - Also, change the log paths after `tee` and `2>` to `output_models/deepseek-3-epochs/train.log` and `.../train.err`.

Then, execute your modified complete command in the terminal.
