
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/albertan017/LLM4Decompile/blob/main/samples/logo-dark.png">
    <img alt="LLM4Decompile" src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/logo-light.png" width=55%>
  </picture>
</p>

<p align="center">
    🚀&nbsp;<a href="#pipeline">Pipeline</a>
    | 📚&nbsp;<a href="#benchmark">Benchmark</a>
    | 🤗&nbsp;<a href="#models">Models</a>
    | 🖥️&nbsp;<a href="#evaluation">Evaluation</a>
    | 📊&nbsp;<a href="#results">Results</a>
</p>

## Updates
* [2025-05-21]: Release [LLM4Decompile-DCBench](https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v1.6), a 1.3 billion-parameter model trained on 10% of the Decompile-Bench, specifically designed to decompile C/C++ code.
* [2025-05-20]: Release [Decompile-Bench](https://huggingface.co/collections/LLM4Binary/decompile-bench-68259091c8d49d0ebd5efda9), contains two million binary-source function pairs for training, and 70K function pairs for evaluation.


## About
* **Decompile-Bench** is the first open-source dataset comprising two million binary-source function pairs condensed from 100 million collected function pairs, i.e., 450GB of binaries compiled from permissively licensed GitHub projects.
* **Decompile-Bench-Eval** includes manually crafted binaries from the well-established HumanEval and MBPP, alongside the compiled GitHub repositories released after 2025 to mitigate data leakage issues.


## Pipeline
<p align="center">
<img src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/dcbench-pipeline.png" alt="image" width="600" height="auto">
</p>

Compile-Trace-Filter framework that automates project compilation, precisely traces function‐level binary-source mappings, and applies robust filters to retain only high-quality pairs.

## Benchmark
[Decompile-Bench](https://huggingface.co/datasets/LLM4Binary/decompile-bench)
contains the following columns:
```
{
"name":"demangled name for the function",
"code":"source code",
"asm":"assembly",
"file":"source code path"
}
```

[Decompile-Bench-Eval](https://huggingface.co/datasets/LLM4Binary/decompile-eval)
contains three splits, huameval, mbpp, and github2025. We also provide a json verison for the data.
They contains the following columns:
```
{
"index":"index of the function", 
"func_name":"demangled name for he function", 
"func_dep":"function dependecies (includes, help functions), or the path to the source code", 
"func":"source code", 
"test":"unit tests for the function, empty for github data", 
"opt":"optimization, O0, O1, O2, O3", 
"language":"language, c or cpp", 
"asm":"assembly", 
"ida_asm":"assembly from ida pro", 
"ida_pseudo":"decompiled results (pseudo code) from ida pro", 
"ghidra_asm":"assembly from ghidra", 
"ghidra_pseudo":"decompiled results (pseudo code) from ghidra"
}
```

## Models
| Model                 | Checkpoint                                                        | Size | HumanEval-Decompile       | Alias |
|-----------------------|-------------------------------------------------------------------|------|---------------------|----------------------|
| **llm4decompile-1.3b-v1.5**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v1.5)   | 1.3B | **16.22%**   | LLM4Decompile-End |
| **llm4decompile-1.3b-v1.6**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v1.6)   | 1.3B | **20.89%**   | LLM4Decompile-DCBench |


## Metrics
* **Re-executability** evaluates whether the decompiled code can execute properly and pass all the predefined test cases.
* **Edit Similarity** based on Levenshtein Distance, this metric captures the minimum number of insertions, deletions, or substitutions needed to turn the generated code into the reference.
For R2I, please refer to the [source project](https://github.com/e0mh4/R2I).
## Requirements
* vllm >= 0.5.2
```
https://docs.vllm.ai/en/v0.5.2/getting_started/installation.html
```

**IMPORTANT**: the libs are required for the compilation, otherwise, the compilation will fail.
```
apt-get update
apt-get install -y libboost-dev libssl-dev
pip install editdistance
```

## Evaluation
* **Re-executability**
```
python3 run_exe_rate.py \
--model_path LLM4Binary/llm4decompile-1.3b-v1.6 \
--dataset_path ./data/humaneval-decompile.json \
--output_path ./data/humaneval
```

* **Edit Similarity**
```
# Note that we assume the decompiled results are stored in the ./data/humaneval
python3 ./metrics/cal_edit_sim.py
```

## Results

<p align="center">
<img src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/dcbench-exe_rate.png" alt="exe" width="800" height="auto">
</p>

<p align="center">
<img src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/dcbench-edit_sim.png" alt="edit" width="800" height="auto">
</p>



