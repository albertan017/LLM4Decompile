# LLM4Decompile

<p align="left">
    📊&nbsp;<a href="#evaluation">Results</a>
    | 🤗&nbsp;<a href="#models">Models</a>
    | 🚀&nbsp;<a href="#quick-start">Quick Start</a>
    | 📚&nbsp;<a href="#humaneval-decompile">HumanEval-Decompile</a>
    | 📎&nbsp;<a href="#citation">Citation</a>
    | 📝&nbsp;<a href="https://arxiv.org/abs/2403.05286">Paper</a>
    | 📝&nbsp;<a href="https://colab.research.google.com/drive/1X5TuUKuNuksGJZz6Cc83KKI0ATBP9q7r?usp=sharing">Colab</a>
</p>

Reverse Engineering: Decompiling Binary Code with Large Language Models

## Updates
* [2024-09-26]: Update a [Colab notebook](https://colab.research.google.com/drive/1X5TuUKuNuksGJZz6Cc83KKI0ATBP9q7r?usp=sharing) to demonstrate the usage of the LLM4Decompile model, including examples for the LLM4Decompile-End and LLM4Decompile-Red models.
* [2024-09-23]: Release [LLM4Decompile-9B-v2](https://huggingface.co/LLM4Binary/llm4decompile-9b-v2), fine-tuned based on [Yi-Coder-9B](https://huggingface.co/01-ai/Yi-Coder-9B), achieved a re-executability rate of **0.6494** on the Decompile benchmark.
* [2024-06-19]: Release [V2](https://huggingface.co/LLM4Binary/llm4decompile-6.7b-v2) series (LLM4Decompile-Ref). V2 (1.3B-22B), building upon **Ghidra**, are trained on 2 billion tokens to **refine** the decompiled pseudo-code from Ghidra. The 22B-V2 version outperforms the 6.7B-V1.5 by an additional 40.1%. Please check the [ghidra folder](https://github.com/albertan017/LLM4Decompile/tree/main/ghidra) for details.
* [2024-05-13]: Release [V1.5](https://huggingface.co/LLM4Binary/llm4decompile-6.7b-v1.5) series (LLM4Decompile-End, directly decompile binary using LLM). V1.5 are trained with a larger dataset (15B tokens) and a maximum token **length of 4,096**, with remarkable  performance (over **100% improvement**) compared to the previous model.
* [2024-03-16]: Add [llm4decompile-6.7b-uo](https://huggingface.co/arise-sustech/llm4decompile-6.7b-uo) model which is trained without prior knowledge of the optimization levels (O0~O3), the average re-executability is around 0.219, performs the best in our models.

## About
* **LLM4Decompile** is the pioneering open-source large language model dedicated to decompilation. Its current version supports decompiling Linux x86_64 binaries, ranging from GCC's O0 to O3 optimization levels, into human-readable C source code. Our team is committed to expanding this tool's capabilities, with ongoing efforts to incorporate a broader range of architectures and configurations.
* **LLM4Decompile-End** focuses on decompiling the binary directly. **LLM4Decompile-Ref** refines the pseudo-code decompiled by Ghidra.


## Evaluation

### Framework
<p align="center">
<img src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/compile-decompile.png" alt="image" width="400" height="auto">
</p>

During compilation, the Preprocessor processes the source code (SRC) to eliminate comments and expand macros or includes. The cleaned code is then forwarded to the Compiler, which converts it into assembly code (ASM). This ASM is transformed into binary code (0s and 1s) by the Assembler. The Linker finalizes the process by linking function calls to create an executable file. Decompilation, on the other hand, involves converting binary code back into a source file. LLMs, being trained on text, lack the ability to process binary data directly. Therefore, binaries must be disassembled by ```Objdump``` into assembly language (ASM) first. It should be noted that binary and disassembled ASM are equivalent, they can be interconverted, and thus we refer to them interchangeably. Finally, the loss is computed between the decompiled code and source code to guide the training. To assess the quality of the decompiled code (SRC'), it is tested for its functionality through test assertions (re-executability).

### Metrics
* **Re-executability** evaluates whether the decompiled code can execute properly and pass all the predefined test cases.

### Benchmarks
* **HumanEval-Decompile** A collection of 164 C functions that exclusively rely on **standard** C libraries.
* **ExeBench** A collection of 2,621 functions drawn from **real** projects, each utilizing user-defined functions, structures, and macros.


### Results

<p align="center">
<img src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/results_paper.png" alt="results" width="800" height="auto">
</p>

## Models
Our LLM4Decompile includes models with sizes between 1.3 billion and 33 billion parameters, and we have made these models available on Hugging Face.

| Model                 | Checkpoint                                                        | Size | Re-executability       | Note |
|-----------------------|-------------------------------------------------------------------|------|---------------------|----------------------|
| llm4decompile-1.3b     | 🤗 [HF Link](https://huggingface.co/arise-sustech/llm4decompile-1.3b)     | 1.3B | 10.6%   |-|
| llm4decompile-6.7b     | 🤗 [HF Link](https://huggingface.co/arise-sustech/llm4decompile-6.7b)     | 6.7B | 21.4%   |-|
| llm4decompile-33b      | 🤗 [HF Link](https://huggingface.co/arise-sustech/llm4decompile-33b)      | 33B  | 21.5%   |-|
| llm4decompile-6.7b-nsp | 🤗 [HF Link](https://huggingface.co/arise-sustech/llm4decompile-6.7b-nsp) | 6.7B | 20.9%   | Note 1 |
| llm4decompile-6.7b-uo  | 🤗 [HF Link](https://huggingface.co/arise-sustech/llm4decompile-6.7b-uo)  | 6.7B | 21.9%   | Note 2 |
| **llm4decompile-1.3b-v1.5**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v1.5)   | 1.3B | **27.3%**   | Note 3 |
| **llm4decompile-6.7b-v1.5**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-6.7b-v1.5)   | 6.7B | **45.4%**   | Note 3 |
| **llm4decompile-1.3b-v2**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v2)   | 1.3B | **46.0%**   | Note 4 |
| **llm4decompile-6.7b-v2**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-6.7b-v2)   | 6.7B | **52.7%**   | Note 4 |
| **llm4decompile-9b-v2**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-9b-v2)   | 9B | **64.9%**  | - |
| **llm4decompile-22b-v2**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-22b-v2)   | 22B | **63.6%**   | Note 4 |

Note 1: The NSP model is trained with assembly code, the average re-executability is around 0.17.

Note 2: The unified optimization (UO) model is trained without prior knowledge of the optimization levels (O0~O3), the average re-executability is around 0.21. The pre-processing of the UO model is slightly different (no prior knowledge of the On), please check the [model page](https://huggingface.co/arise-sustech/llm4decompile-6.7b-uo#3-how-to-use).

Note 3: V1.5 series are trained with a larger dataset (15B tokens) and a maximum token size of 4,096, with remarkable performance (over 100% improvement) compared to the previous model.

Note 4: V2 series are built upon **Ghidra** and trained on 2 billion tokens to **refine** the decompiled pseudo-code from Ghidra. Check [ghidra folder](https://github.com/albertan017/LLM4Decompile/tree/main/ghidra) for details.

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

Here is an example of how to use our model (Revised for V1.5. For previous models, please check the corresponding model page at HF).
Note: **Replace** func0 with the function name you want to decompile.

**Preprocessing:** Compile the C code into binary, and disassemble the binary into assembly instructions.
```python
import subprocess
import os

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
        if '<'+'func0'+'>:' not in asm: #IMPORTANT replace func0 with the function name
            raise ValueError("compile fails")
        asm = '<'+'func0'+'>:' + asm.split('<'+'func0'+'>:')[-1].split('\n\n')[0] #IMPORTANT replace func0 with the function name
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

<FUNCTION_NAME>:\nOPERATIONS\nOPERATIONS\n

Typical assembly instructions may look like this:
```
<func0>:
endbr64
lea    (%rdi,%rsi,1),%eax
retq
```


**Decompilation:** Use LLM4Decompile to translate the assembly instructions into C:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = 'LLM4Binary/llm4decompile-6.7b-v1.5' # V1.5 Model
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

## HumanEval-Decompile
Data are stored in ``llm4decompile/decompile-eval/decompile-eval-executable-gcc-obj.json``, using JSON list format. There are 164*4 (O0, O1, O2, O3) samples, each with five keys:

*   ``task_id``: indicates the ID of the problem.
*   ``type``: the optimization stage, is one of [O0, O1, O2, O3].
*   ``c_func``: C solution for HumanEval problem. 
*   ``c_test``: C test assertions.
*   ``input_asm_prompt``: assembly instructions with prompts, can be derived as in our [preprocessing example](https://github.com/albertan017/LLM4Decompile?tab=readme-ov-file#quick-start).

Please check the [evaluation scripts](https://github.com/albertan017/LLM4Decompile/tree/main/evaluation).

## On Going
* Larger training dataset with the cleaning process. (done:2024.05.13)
* Support for popular languages/platforms and settings.
* Support for executable binaries. (done:2024.05.13)
* Integration with decompilation tools (e.g., Ghidra, Rizin)

## License
This code repository is licensed under the MIT and DeepSeek License.

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=albertan017/LLM4Decompile&type=Timeline)](https://star-history.com/#albertan017/LLM4Decompile&Timeline)
