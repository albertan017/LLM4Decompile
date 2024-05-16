import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import torch
import re
import json
from tqdm import tqdm, trange
os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument('--model_path',type=str,default='LLM4Binary/llm4decompile-6.7b-v1.5',required=False)
parser.add_argument('--data_path',type=str,default='../decompile-eval/decompile-eval-executable-gcc-obj.json',required=False)

args = parser.parse_args()

def evaluate_func(c_func,c_test,c_func_decompile):
    flag_compile = 0
    flag_run = 0
    c_include = ''
    for line in c_func.split('\n'):
        if '#include' in line:
            c_include += line+'\n'
            c_func = c_func.replace(line, '')
    for line in c_test.split('\n'):
        if '#include' in line:
            c_include += line+'\n'
            c_test = c_test.replace(line, '')
    c_combine = c_include + '\n' + c_func_decompile + '\n' + c_test
    c_onlyfunc = c_include + '\n' + c_func_decompile

    # Define the C file and executable names
    c_file = 'combine.c'
    executable = 'combine'
    if os.path.exists(executable):
        os.remove(executable)

    c_file_onlyfunc = 'onlyfunc.c'
    executable_onlyfunc = 'onlyfunc'
    if os.path.exists(executable):
        os.remove(executable_onlyfunc)

    with open(c_file,'w') as f:
        f.write(c_combine)
    with open(c_file_onlyfunc,'w') as f:
        f.write(c_onlyfunc)

    # Compile the C program to an assembly
    compile_command = f'gcc -S {c_file_onlyfunc} -o {executable_onlyfunc} -lm'
    try:
        subprocess.run(compile_command, shell=True, check=True)
        flag_compile = 1
    except:
        return flag_compile, flag_run

    # Compile the C program to an executable
    compile_command = f'gcc {c_file} -o {executable} -lm'
    try:
        subprocess.run(compile_command, shell=True, check=True)
        flag_compile = 1
    except:
        return flag_compile, flag_run

    # Run the compiled executable
    run_command = f'./{executable}'
    try:
        process = subprocess.run(run_command, shell=True, check=True,capture_output=True, timeout=5)
        flag_run = 1
    except subprocess.CalledProcessError as e:
        pass
    except Exception as e:
        pass
    return flag_compile, flag_run

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).cuda()
print('Model Loaded!')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id

OPT = ["O0", "O1", "O2", "O3"]  # Optimization states
with open(args.data_path,'r') as f:
    data_all = json.load(f)
NUM = int(len(data_all)/4)
num_compile = {"O0":0, "O1":0, "O2":0, "O3":0}
num_run = {"O0":0, "O1":0, "O2":0, "O3":0}

for idx in trange(len(data_all)):
    c_func = data_all[idx]['c_func']
    c_test = data_all[idx]['c_test']
    input_asm_prompt = data_all[idx]['input_asm_prompt']
    opt_state = data_all[idx]['type']
    before = f"# This is the assembly code with {opt_state} optimization:\n"#prompt
    after = "\n# What is the source code?\n"#prompt
    input_asm_prompt = before+input_asm_prompt.strip()+after
    inputs = tokenizer(input_asm_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    c_func_decompile = tokenizer.decode(outputs[0][len(inputs[0]):-1])
    flag_compile,flag_run = evaluate_func(c_func,c_test,c_func_decompile)
    num_compile[opt_state]+=flag_compile
    num_run[opt_state]+=flag_run
with open('results.txt','a') as f:
    for opt_state in num_compile.keys():
        f.write('model:{},opt:{},compile rate:{:.4f},run_rate:{:.4f}\n'.format(args.model_path,opt_state,num_compile[opt_state]/NUM,num_run[opt_state]/NUM))

