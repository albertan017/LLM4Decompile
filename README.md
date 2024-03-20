# LLM4Decompile

<p align="left">
    📊&nbsp;<a href="#evaluation-results">Results</a>
    | 🤗&nbsp;<a href="#models">Models</a>
    | 🚀&nbsp;<a href="#quick-start">Quick Start</a>
    | 📚&nbsp;<a href="#decompile-eval">Decompile-Eval</a>
    | 📎&nbsp;<a href="#citation">Citation</a>
    | 📝&nbsp;<a href="https://arxiv.org/abs/2403.05286">Paper</a>
</p>

Reverse Engineering: Decompiling Binary Code with Large Language Models

## Updates
* [2023-03-16] : Add [llm4decompile-6.7b-uo](https://huggingface.co/arise-sustech/llm4decompile-6.7b-uo) model which is trained without prior knowledge of the optimization levels (O0~O3), the average re-executability is arond 0.21.

## About
* **LLM4Decompile** is the pioneering open-source large language model dedicated to decompilation. Its current version supports decompiling Linux x86_64 binaries, ranging from GCC's O0 to O3 optimization levels, into human-readable C source code. Our team is committed to expanding this tool's capabilities, with ongoing efforts to incorporate a broader range of architectures and configurations.
* **Decompile-Eval** is the first decompilation benchmark focuses on assessing the re-compilability and re-executability aspects of decompiled code. It is the C language adaptation of the HumanEval dataset, provides a suite of C solutions and assertions in evaluating the practical utility of decompiled code.


## Evaluation Results
### Metrics
**Re-compilability** and **re-executability** serve as critical indicators in validating the effectiveness of a decompilation process. When decompiled code can be recompiled, it provides strong evidence of syntactic integrity. It ensures that the decompiled code is not just readable, but also adheres to the structural and syntactical standards expected by the compiler. 
However, syntax alone does not guarantee semantic equivalence to the original pre-compiled program. Re-executability provides this critical measure of semantic correctness. By re-compiling the decompiled output and running the test cases, we assess if the decompilation preserved the program logic and behavior.
Together, re-compilability and re-executability indicate syntax recovery and semantic preservation - both essential for usable and robust decompilation.

### Results
![Alt text](https://github.com/albertan017/LLM4Decompile/blob/main/samples/results_decompile.png)



## Models
Our LLM4Decompile includes models with sizes between 1.3 billion and 33 billion parameters, and we have made these models available on Hugging Face.

[llm4decompile-1.3b](https://huggingface.co/arise-sustech/llm4decompile-1.3b)

[llm4decompile-6.7b](https://huggingface.co/arise-sustech/llm4decompile-6.7b)

[llm4decompile-33b](https://huggingface.co/arise-sustech/llm4decompile-33b)

[llm4decompile-6.7b-nsp](https://huggingface.co/arise-sustech/llm4decompile-6.7b-nsp)

[llm4decompile-6.7b-uo](https://huggingface.co/arise-sustech/llm4decompile-6.7b-uo)

Note: The NSP model is trained with assembly code, the average re-executability is arond 0.17.

Note: The unified optimization (UO) model is trained without prior knowledge of the optimization levels (O0~O3), the average re-executability is arond 0.21. The pre-processing of UO model is slightly different (no prior knowledge of the On), please check the [model page](https://huggingface.co/arise-sustech/llm4decompile-6.7b-uo#3-how-to-use).


## Quick Start
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
    with open(output_file+'.s') as f:#asm file
        asm= f.read()
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
    outputs = model.generate(**inputs, max_new_tokens=500)
c_func_decompile = tokenizer.decode(outputs[0][len(inputs[0]):-1])
```

## Decompile-Eval
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

## On Going
LLM4Binary: We plan to include larger dataset to pre-train the model with assembly code and C code.

Decompiler-ALL: Support popular languages/platforms and settings (e.g., decompile multiple functions).

## License
This code repository is licensed under the MIT License.

## Contact

If you have any questions, please raise an issue.

## Thoughts

The conversation about the language model decompiler that took place on [Reddit](https://www.reddit.com/r/MachineLearning/comments/123asbg/d_can_we_train_a_decompiler/) roughly a year ago was quite fascinating to us.

## Citation
```
@misc{tan2024llm4decompile,
      title={LLM4Decompile: Decompiling Binary Code with Large Language Models}, 
      author={Hanzhuo Tan and Qi Luo and Jing Li and Yuqun Zhang},
      year={2024},
      eprint={2403.05286},
      archivePrefix={arXiv},
      primaryClass={cs.PL}
}
```
