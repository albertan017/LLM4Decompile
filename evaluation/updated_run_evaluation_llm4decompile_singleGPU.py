import subprocess
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='LLM4Binary/llm4decompile-6.7b-v1.5')
parser.add_argument('--data_path', type=str, default='../decompile-eval/decompile-eval-executable-gcc-obj.json')
args = parser.parse_args()

def evaluate_func(c_func, c_test, c_func_decompile):
    # Extract includes and combine C code
    includes = '\n'.join(line for line in c_func.splitlines() if '#include' in line)
    c_func = re.sub(r'#include.*', '', c_func)
    c_test = re.sub(r'#include.*', '', c_test)
    c_combine = f"{includes}\n{c_func_decompile}\n{c_test}"
    c_onlyfunc = f"{includes}\n{c_func_decompile}"

    # Write combined C files
    with open('combine.c', 'w') as f:
        f.write(c_combine)
    with open('onlyfunc.c', 'w') as f:
        f.write(c_onlyfunc)

    # Compile and run
    flags = {'compile': 0, 'run': 0}
    compile_commands = [
        ('-S', 'onlyfunc.c', 'onlyfunc'),
        ('', 'combine.c', 'combine')
    ]

    for flag, c_file, executable in compile_commands:
        if os.path.exists(executable):
            os.remove(executable)
        try:
            subprocess.run(f'gcc {flag} {c_file} -o {executable} -lm', shell=True, check=True)
            flags['compile'] = 1
        except subprocess.CalledProcessError:
            return flags['compile'], flags['run']

    try:
        subprocess.run(f'./combine', shell=True, check=True, capture_output=True, timeout=5)
        flags['run'] = 1
    except subprocess.CalledProcessError:
        pass

    return flags['compile'], flags['run']

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id

# Processing data
with open(args.data_path, 'r') as f:
    data_all = json.load(f)

NUM = len(data_all) // 4
num_compile = {opt: 0 for opt in ["O0", "O1", "O2", "O3"]}
num_run = {opt: 0 for opt in ["O0", "O1", "O2", "O3"]}

for entry in trange(len(data_all)):
    c_func = entry['c_func']
    c_test = entry['c_test']
    input_asm_prompt = entry['input_asm_prompt']
    opt_state = entry['type']

    # Prepare and process prompt
    input_asm_prompt = (f"# This is the assembly code with {opt_state} optimization:\n"
                        f"{input_asm_prompt.strip()}\n# What is the source code?\n")
    inputs = tokenizer(input_asm_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    c_func_decompile = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Evaluate and count results
    flag_compile, flag_run = evaluate_func(c_func, c_test, c_func_decompile)
    num_compile[opt_state] += flag_compile
    num_run[opt_state] += flag_run

# Write results
with open('results.txt', 'a') as f:
    for opt_state in num_compile:
        f.write(f'model:{args.model_path},opt:{opt_state},compile rate:{num_compile[opt_state]/NUM:.4f},run rate:{num_run[opt_state]/NUM:.4f}\n')
