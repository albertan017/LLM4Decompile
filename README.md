# LLM4Decompile
Reverse Engineering: Decompiling Binary Code with Large Language Models

## 1. Introduction of LLM4Decompile and Decompile-Eval
Our objective is to create and release the first open-source LLM dedicated to decompilation, and to assess its capabilities by constructing the first decompilation benchmark focused on re-compilability and re-executable. 

We start by compiling a million C code samples from AnghaBench into assembly code using GCC with different configurations, forming a dataset of assembly-source pairs in 4 billion tokens. We then finetune the DeepSeek-Coder model, a leading-edge code LLM, using this dataset. Followed by constructing the evaluation benchmark, Decompile-Eval, based on HumanEval questions and test samples. Specifically, we formulate the evaluation from two perspectives: whether the decompiled code can recompile successfully, and whether it passes all assertions in the test cases. 

Figure 1 presents the steps involved in our decompilation evaluation. First, the source code (denoted as src) is compiled by the GCC compiler with specific parameters, such as optimization levels, to produce the executable binary. This binary is then disassembled into assembly language (asm) using the objdump tool. The assembly instructions are subsequently decompiled to reconstruct the source code in a format that's readable to humans (noted as src'). To assess the quality of the decompiled code (src'), it is tested for its ability to be recompiled with the original GCC compiler (re-compilability) and for its functionality through test assertions (re-executability).
<p align="center">
<img src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/pipeline.png" alt="image" width="300" height="auto">
</p>

## 2. Evaluation Results
### Metrics
**Re-compilability** and **re-executability** serve as critical indicators in validating the effectiveness of a decompilation process. When decompiled code can be recompiled, it provides strong evidence of syntactic integrity. It ensures that the decompiled code is not just readable, but also adheres to the structural and syntactical standards expected by the compiler. 
However, syntax alone does not guarantee semantic equivalence to the original pre-compiled program. Re-executability provides this critical measure of semantic correctness. By re-compiling the decompiled output and running the test cases, we assess if the decompilation preserved the program logic and behavior.
Together, re-compilability and re-executability indicate syntax recovery and semantic preservation - both essential for usable and robust decompilation.

### Results
![Alt text](https://github.com/albertan017/LLM4Decompile/blob/main/samples/results_decompile.png)



## 3. How to Use The Model
Our LLM4Decompile includes models with sizes between 1.3 billion and 33 billion parameters, and we have made these models available on Hugging Face.

[llm4decompile-1.3b](https://huggingface.co/arise-sustech/llm4decompile-1.3b)

[llm4decompile-6.7b](https://huggingface.co/arise-sustech/llm4decompile-6.7b)

[llm4decompile-33b](https://huggingface.co/arise-sustech/llm4decompile-33b)

[llm4decompile-6.7b-nsp](https://huggingface.co/arise-sustech/llm4decompile-6.7b-nsp)

Note: The NSP model is trained with assembly code, the average re-executability is arond 0.17.

Here give an example of how to use our model.

**Preprocessing:** compile the C code into binary, disassemble the binary into assembly instructions.
```python
import subprocess
import os
import re

digit_pattern = r'\b0x[a-fA-F0-9]+\b'# binary codes in Hexadecimal
zeros_pattern = r'^0+\s'#0s
OPT = ["O0", "O1", "O2", "O3"]
fileName = 'path/to/file'
with open(fileName+'.c','r') as f:#original file
    c_func = f.read()
for opt_state in OPT:
    output_file = fileName +'_' + opt_state
    input_file = fileName+'.c'
    compile_command = f'gcc -c -o {output_file}.o {input_file} -{opt_state} -lm'#compile the code with GCC on Linux
    subprocess.run(compile_command, shell=True, check=True)
    compile_command = f'objdump -d {output_file}.o > {output_file}.s'#disassemble the binary file into assembly instructions
    subprocess.run(compile_command, shell=True, check=True)
    
    input_asm = ''
    asm = read_file(output_file+'.s')
    asm = asm.split('Disassembly of section .text:')[-1].strip()
    for tmp in asm.split('\n'):
        tmp_asm = tmp.split('\t')[-1]#remove the binary code
        tmp_asm = tmp_asm.split('#')[0].strip()#remove the comments
        input_asm+=tmp_asm+'\n'
    input_asm = re.sub(zeros_pattern, '', input_asm)
    before = f"# This is the assembly code with {opt_state} optimization:\n"#prompt
    after = "\n# What is the source code?\n"#prompt
    input_asm_prompt = before+input_asm.strip()+after
    with open(fileName +'_' + opt_state +'.asm','w',encoding='utf-8') as f:
        f.write(input_asm_prompt)
```

**Decompilation:** use LLM4Decompile to translate the assembly instructions into C:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = 'arise-sustech/llm4decompile-1.3b'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).cuda()

with open(fileName +'_' + opt_state +'.asm','r') as f:#original file
    asm_func = f.read()
inputs = tokenizer(asm_func, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
c_func_decompile = tokenizer.decode(outputs[0][len(inputs[0]):-1])
```

## 4. How to use Decompile-Eval
Data are stored in ``llm4decompile/decompile-eval/decompile-eval.json``, using JSON list format. There are 164*4 (O0, O1, O2, O3) samples, each with five keys:

*   ``task_id``: indicates the ID of the problem.
*   ``type``: the optimization stage, is one of [O0, O1, O2, O3].
*   ``c_func``: C solution for HumanEval problem. 
*   ``c_test``: C test assertions.
*   ``input_asm_prompt``: assembly instructions with prompts, can be derived as in our [preprocessing example](https://github.com/albertan017/LLM4Decompile/blob/main/README.md#3-how-to-use-the-model).

To run the evaluation on single GPU and single process:
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

## 5. On Going
LLM4Binary: We plan to include larger dataset to pre-train the model with assembly code and C code.

## 6. License
This code repository is licensed under the MIT License.

## 7. Contact

If you have any questions, please raise an issue.
