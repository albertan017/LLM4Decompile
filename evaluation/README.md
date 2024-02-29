To run the evaluation on Single GPU and single process:
```bash
cd LLM4Decompile
python ./evaluation/run_evaluation_llm4decompile_singleGPU.py
```

To run the evaluation using TGI (10x faster, support multiple GPUs and multi-process):
First, please install the text-generation-inference following the official [link](https://github.com/huggingface/text-generation-inference)
```bash
git clone https://github.com/albertan017/LLM4Decompile.git
cd LLM4Decompile
pip install -r requirements.txt

# Before run the evaluation script, plase update the model_path to your local mdoel path.
bash ./scripts/run_evaluation_llm4decompile.sh
```
