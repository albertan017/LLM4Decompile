## Updates
* [2024-05-16]: Please use ``decompile-eval-executable-gcc-obj.json``. The source codes are compiled into executable binaries and disassembled into assembly instructions.
* [2024-04-10]: Add vllm evaluation script.


To run the evaluation using [vLLM](https://github.com/vllm-project/vllm) (**Recommended**)
```bash
pip install -r requirements.txt
```
To use the flash-attention backend to speed up the interface, you can install it via `pip install flash-attn`.
```bash
cd evaluation
# Before running the evaluation script, please update the model_path to your local model path.
python run_evaluation_llm4decompile_vllm.py \
  --model_path LLM4Binary/llm4decompile-6.7b-v1.5 \
  --testset_path ../decompile-eval/decompile-eval-executable-gcc-obj.json \
  --gpus 8 \
  --max_total_tokens 8192 \
  --max_new_tokens 512 \
  --repeat 1 \
  --num_workers 16 \
  --gpu_memory_utilization 0.82 \
  --temperature 0 
```

---
To run the evaluation on single GPU and single process: (legacy, not updated)
```bash
cd LLM4Decompile
python ./evaluation/run_evaluation_llm4decompile_singleGPU.py
```
---
To run the evaluation using TGI (10x faster, support multiple GPUs and multi-process): (legacy, not updated)
First, please install the text-generation-inference following the official [link](https://github.com/huggingface/text-generation-inference)
```bash
git clone https://github.com/albertan017/LLM4Decompile.git
cd LLM4Decompile
pip install -r requirements.txt

# Before running the evaluation script, please update the model_path to your local model path.
bash ./scripts/run_evaluation_llm4decompile.sh
```
---


