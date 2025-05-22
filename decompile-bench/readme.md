
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
    | 📎&nbsp;<a href="#quick-start">Quick Start</a>
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

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X5TuUKuNuksGJZz6Cc83KKI0ATBP9q7r?usp=sharing)

**Setup:** Please use the script below to install the necessary environment.
```
git clone https://github.com/albertan017/LLM4Decompile.git
cd LLM4Decompile
conda create -n 'llm4decompile' python=3.9 -y
conda activate llm4decompile
pip install -r requirements.txt
```

Here is an example of how to use our model (For previous models, please check the corresponding model page at HF).
Note: **Replace the "func0" with the function name you want to decompile**.

**Preprocessing:** Compile the C code into binary, and disassemble the binary into assembly instructions.
```python
import subprocess
import os
func_name = 'func0'
OPT = ["O0", "O1", "O2", "O3"]
fileName = 'samples/sample' #'path/to/file'
for opt_state in OPT:
    output_file = fileName +'_' + opt_state
    input_file = fileName+'.c'
    compile_command = f'gcc -o {output_file}.o {input_file} -{opt_state} -lm'#compile the code with GCC on Linux
    subprocess.run(compile_command, shell=True, check=True)
    compile_command = f'objdump -d {output_file}.o > {output_file}.s'#disassemble the binary file into assembly instructions
    subprocess.run(compile_command, shell=True, check=True)
    
    input_asm = ''
    with open(output_file+'.s') as f:#asm file
        asm= f.read()
        if '<'+func_name+'>:' not in asm: #IMPORTANT replace func0 with the function name
            raise ValueError("compile fails")
        asm = func_name+':' + asm.split('<'+func_name+'>:')[-1].split('\n\n')[0] #IMPORTANT replace func0 with the function name
        asm_clean = ""
        asm_sp = asm.split("\n")
        for tmp in asm_sp:
            if len(tmp.split("\t"))<3 and '00' in tmp:
                continue
            idx = min(
                len(tmp.split("\t")) - 1, 2
            )
            tmp_asm = "\t".join(tmp.split("\t")[idx:])  # remove the binary code
            tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments
            asm_clean += tmp_asm + "\n"
    input_asm = asm_clean.strip()
    before = f"# This is the assembly code:\n"#prompt
    after = "\n# What is the source code?\n"#prompt
    input_asm_prompt = before+input_asm.strip()+after
    with open(fileName +'_' + opt_state +'.asm','w',encoding='utf-8') as f:
        f.write(input_asm_prompt)
```

Assembly instructions should be in the format:
```
FUNCTION_NAME:
OPERATIONS
OPERATIONS
```

Typical assembly instructions may look like this:
```
func0:
endbr64
lea    (%rdi,%rsi,1),%eax
retq
```


**Decompilation:** Use LLM4Decompile to translate the assembly instructions into C:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = 'LLM4Binary/llm4decompile-1.3b-v1.6' # V1.6 Model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).cuda()

with open(fileName +'_' + OPT[0] +'.asm','r') as f:#optimization level O0
    asm_func = f.read()
inputs = tokenizer(asm_func, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=2048)### max length to 4096, max new tokens should be below the range
c_func_decompile = tokenizer.decode(outputs[0][len(inputs[0]):-1])

with open(fileName +'.c','r') as f:#original file
    func = f.read()

print(f'original function:\n{func}')# Note we only decompile one function, where the original file may contain multiple functions
print(f'decompiled function:\n{c_func_decompile}')
```

