#!/bin/bash

workspace="$(pwd)"
python $workspace/evaluation/run_evaluation_llm4decompile.py \
    --model_path llm4decompile-1.3b \
    --max_new_tokens 512 \
    --testset_path $workspace/decompile-eval/decompile-eval.json \
    --repeat 1 \
    --dtype bfloat16 \
    --port 8080 \
    --max_input_len 8000 \
    --max_total_tokens 8512 \
    --max_batch_prefill_tokens 36000 \
    --num_shards 4 \
    --num_workers 8 \
