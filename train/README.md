## How to Fine-tune LLM4Decompile (Based on Ghidra Pseudo-Code)

We provide script `finetune.py`, adapted from the [deepseek-coder](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/finetune/finetune_deepseekcoder.py) repository.

The script supports the training with [DeepSpeed](https://github.com/microsoft/DeepSpeed). You need install required packages by:

```bash
pip install -r requirements.txt
```

If you want to leverage FlashAttention to accelerate training, install it via:
```bash
pip install flash-attn
```

Please download the [decompile-ghidra-100k](https://huggingface.co/datasets/LLM4Binary/decompile-ghidra-100k) dataset to your workspace, and process it into JSON format. 
Each line is a json-serialized string with two required fields `instruction` and `output`.

After data preparation, you can use the sample shell script to finetune llm4decompile model. 
Remember to specify `DATA_PATH`, `OUTPUT_PATH`.

```bash
WORKSPACE="/workspace"
DATA_PATH="${WORKSPACE}/decompile-ghidra-100k.json"
OUTPUT_PATH="${WORKSPACE}/output_models/llm4decompile-ref"
MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"

CUDA_VISIBLE_DEVICES=0 deepspeed finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 2 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --use_flash_attention \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --warmup_ratio 0.025 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --bf16 True
```


## Simple demo on constructing the training data (Based on Objdump assembly). Note we use ExeBench as our final dataset.

Before compiling, please clone the [AnghaBench](https://github.com/brenocfg/AnghaBench) dataset.

```bash
git clone https://github.com/brenocfg/AnghaBench
```

Then use the following script to compile AnghaBench:
```bash
python compile.py --root Anghabench_path --output AnghaBench_compile.jsonl
```
