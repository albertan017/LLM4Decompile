import json
import tempfile
import os
import subprocess
from tqdm import tqdm,trange

OPT = ["O0", "O1", "O2", "O3"]
timeout_duration = 10

ghidra_path = "./ghidra_11.0.3_PUBLIC/support/analyzeHeadless"#path to the headless analyzer, change the path accordingly
postscript = "./decompile.py"#path to the decompiler helper function, change the path accordingly
project_path = "."#path to temp folder for analysis, change the path accordingly
project_name = "tmp_ghidra_proj"
func_path = "../samples/sample.c"#path to c code for compiling and decompiling, change the path accordingly
fileName = "sample"

with tempfile.TemporaryDirectory() as temp_dir:
    pid = os.getpid()
    asm_all = {}
    for opt in [OPT[0]]:
        executable_path = os.path.join(temp_dir, f"{pid}_{opt}.o")
        cmd = f'gcc -{opt} -o {executable_path} {func_path} -lm'
        subprocess.run(
        cmd.split(' '),
        check=True,
        stdout=subprocess.DEVNULL,  # Suppress stdout
        stderr=subprocess.DEVNULL,  # Suppress stderr
        timeout=timeout_duration,
        )

        output_path = os.path.join(temp_dir, f"{pid}_{opt}.c")
        command = [
            ghidra_path,
            temp_dir,
            project_name,
            "-import", executable_path,
            "-postScript", postscript, output_path,
            "-deleteProject",  # WARNING: This will delete the project after analysis
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        with open(output_path,'r') as f:
            c_decompile = f.read()
        c_func = []
        flag = 0
        for line in c_decompile.split('\n'):
            if "Function: func0" in line:#**Replace** func0 with the function name you want to decompile.
                flag = 1
                c_func.append(line)
                continue
            if flag:
                if '// Function:' in line:
                    if len(c_func) > 1:
                        break
                c_func.append(line)
        if flag == 0:
            raise ValueError('bad case no function found')
        for idx_tmp in range(1,len(c_func)):##########remove the comments
            if 'func0' in c_func[idx_tmp]:
                break
        c_func = c_func[idx_tmp:]
        input_asm = '\n'.join(c_func).strip()

        before = f"# This is the assembly code:\n"#prompt
        after = "\n# What is the source code?\n"#prompt
        input_asm_prompt = before+input_asm.strip()+after
        with open(fileName +'_' + opt +'.pseudo','w',encoding='utf-8') as f:
            f.write(input_asm_prompt)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = 'LLM4Binary/llm4decompile-6.7b-v2' # V2 Model

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()

with open(fileName +'_' + OPT[0] +'.pseudo','r') as f:#optimization level O0
    asm_func = f.read()
inputs = tokenizer(asm_func, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=2048)### max length to 4096, max new tokens should be below the range
c_func_decompile = tokenizer.decode(outputs[0][len(inputs[0]):-1])

with open(fileName +'_' + OPT[0] +'.pseudo','r') as f:#original file
    func = f.read()

print(f'pseudo function:\n{func}')# Note we only decompile one function, where the original file may contain multiple functions
print(f'refined function:\n{c_func_decompile}')
