# LLM4Decompile
Reverse Engineering: Decompiling Binary Code with Large Language Models

### 1. Introduction of LLM4Decompile

LLM4Decompile aims to decompile x86 assembly instructions into C. It is finetuned from Deepseek-Coder on 2B tokens of assembly-C pairs compiled from AnghaBench.


### 2. Evaluation Results
| Model              | Re-compilability |           |           |           |           | Re-executability |           |           |           |           |
|--------------------|:----------------:|:---------:|:---------:|:---------:|:---------:|:----------------:|-----------|-----------|-----------|:---------:|
| opt-level          | O0               | O1        | O2        | O3        | Avg.      | O0               | O1        | O2        | O3        | Avg.      |
| GPT4               | 0.92             | 0.94      | 0.88      | 0.84      | 0.895     | 0.1341           | 0.1890    | 0.1524    | 0.0854    | 0.1402    |
| DeepSeek-Coder-33B |   0.0659         |   0.0866  |   0.1500  |   0.1463  |   0.1122  |   0.0000         |   0.0000  |   0.0000  |   0.0000  |   0.0000  |
| LLM4Decompile-1b   |   0.8780         |   0.8732  |   0.8683  |   0.8378  |   0.8643  |   0.1573         |   0.0768  |   0.1000  |   0.0878  |   0.1055  |
| LLM4Decompile-6b   |   0.8817         |   0.8951  |   0.8671  |   0.8476  |   0.8729  |   0.3000         |   0.1732  |   0.1988  |   0.1841  |   0.2140  |
| LLM4Decompile-33b  |   0.8134         |   0.8195  |   0.8183  |   0.8305  |   0.8204  |   0.3049         |   0.1902  |   0.1817  |   0.1817  |   0.2146  |



### 3. How to Use
Here give an example of how to use our model.
First compile the C code into binary, disassemble the binary into assembly instructions:
```python
import subprocess
import os
import re

digit_pattern = r'\b0x[a-fA-F0-9]+\b'# hex lines
zeros_pattern = r'^0+\s'#0s
OPT = ["O0", "O1", "O2", "O3"]
before = f"# This is the assembly code with {opt_state} optimization:\n"
after = "\n# What is the source code?\n"
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
    
    input_asm_prompt = before+input_asm.strip()+after
    with open(fileName +'_' + opt_state +'.asm','w',encoding='utf-8') as f:
        f.write(input_asm_prompt)
```

Then use LLM4Decompile to translate the assembly instructions into C:
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

### 4. License
This code repository is licensed under the MIT License.

### 5. Contact

If you have any questions, please raise an issue.
