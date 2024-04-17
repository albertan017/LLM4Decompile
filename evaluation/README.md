## Updates
* [2023-04-10]: Add vllm evaluation script.

---
To run the evaluation on single GPU and single process:
```bash
cd LLM4Decompile
python ./evaluation/run_evaluation_llm4decompile_singleGPU.py
```
---
To run the evaluation using TGI (10x faster, support multiple GPUs and multi-process):
First, please install the text-generation-inference following the official [link](https://github.com/huggingface/text-generation-inference)
```bash
git clone https://github.com/albertan017/LLM4Decompile.git
cd LLM4Decompile
pip install -r requirements.txt

# Before run the evaluation script, plase update the model_path to your local mdoel path.
bash ./scripts/run_evaluation_llm4decompile.sh
```
---
To run the evaluation using [vLLM](https://github.com/vllm-project/vllm)
```bash
pip install -r requirements.txt
cd evaluation
# Before run the evaluation script, plase update the model_path to your local mdoel path.
python run_evaluation_llm4decompile_vllm.py \
  --model_path arise-sustech/llm4decompile-1.3b \
  --testset_path ../decompile-eval/decompile-eval.json \
  --gpus 8 \
  --max_total_tokens 8192 \
  --max_new_tokens 512 \
  --repeat 1 \
  --num_workers 16 \
  --gpu_memory_utilization 0.82 \
  --temperature 0 
```

