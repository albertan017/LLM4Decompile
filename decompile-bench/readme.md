
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/albertan017/LLM4Decompile/blob/main/samples/logo-dark.png">
    <img alt="LLM4Decompile" src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/logo-light.png" width=55%>
  </picture>
</p>

<p align="left">
    📊&nbsp;<a href="#evaluation">Results</a>
    | 🤗&nbsp;<a href="#models">Models</a>
    | 🚀&nbsp;<a href="#collection">Collection</a>
    | 📚&nbsp;<a href="#benchmark">Benchmark</a>
</p>

## Updates
* [2025-05-20]: Release [decompile-bench](https://huggingface.co/collections/LLM4Binary/decompile-bench-68259091c8d49d0ebd5efda9), contains two million binary-source function pairs for training, and 70K function pairs for evaluation.


## About
* **Decompile-Bench** is the first open-source dataset comprising two million binary-source function pairs condensed from 100 million collected function pairs, i.e., 450GB of binaries compiled from permissively licensed GitHub projects.
* **Decompile-Bench-Eval** includes manually crafted binaries from the well-established HumanEval and MBPP, alongside the compiled GitHub repositories released after 2025 to mitigate data leakage issues.


## Collection
<p align="center">
<img src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/dcbench-pipeline.png" alt="image" width="600" height="auto">
</p>

Compile-Trace-Filter framework that automates project compilation, precisely traces function‐level binary-source mappings, and applies robust filters to retain only high-quality pairs.

## Metrics
* **Re-executability** evaluates whether the decompiled code can execute properly and pass all the predefined test cases.
* **Edit Similarity** based on Levenshtein Distance, this metric captures the minimum number of insertions, deletions, or substitutions needed to turn the generated code into the reference.

## Evaluation

## Models
LLM4Binary/llm4decompile-1.3b-v1.6

## Requirements
* vllm >= 0.5.2
```
https://docs.vllm.ai/en/v0.5.2/getting_started/installation.html
```

```
apt-get update
apt-get install -y libboost-dev libssl-dev
pip install editdistance
```

**IMPORTANT**: the libs are required for the compilation, otherwise, the compilation will fail.

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
