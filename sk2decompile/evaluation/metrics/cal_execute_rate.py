#apt-get update
#apt-get install -y libboost-dev libssl-dev
import subprocess
import os
import json
# from loguru import logger
import traceback
from argparse import ArgumentParser
import sys
from tqdm import tqdm
import multiprocessing
import tempfile
import numpy as np

# logger.add(sys.stdout, colorize=False, format="{time} {level} {message}")

current_dir = os.path.dirname(os.path.abspath(__file__))

def execute_rate(func_dep, func, func_test=None, timeout=10, language='cpp', opt="-O0"):
    flag_exe = 0
    flag_comp = 0

    if func_test!=None:
        func_exe = func_dep + "\n" + func + "\n" + func_test
    else:
        func_comp = func_dep + "\n" + func  # no test, only compile

    with tempfile.TemporaryDirectory() as temp_dir:
        pid = os.getpid()
        file_exe = os.path.join(temp_dir, f"exe_{pid}.c")
        binary_exe = os.path.join(temp_dir, f"exe_{pid}")
        with open(file_exe, "w") as f:
            f.write(func_exe)
        # Compile the C program to an executable
        if language == 'cpp':

            if func_test!=None:
                compile_command = ["g++", opt, '-std=c++17', file_exe, "-o", binary_exe, "-lm", "-lcrypto"]
            else:
                compile_command = ["g++", opt, '-S', '-std=c++17', file_exe, "-o", binary_exe, "-lm", "-lcrypto"]    # only compile
        else:
            if func_test!=None:
                compile_command = ["gcc", opt, file_exe, "-o", binary_exe, "-lm"]
            else:
                compile_command = ["gcc", opt, '-S', file_exe, "-o", binary_exe, "-lm"]    # only compile

        try:
            subprocess.run(compile_command, check=True, timeout=timeout)
            flag_comp = 1
        except:
            return flag_comp, flag_exe

        if func_test==None:
            return flag_comp, flag_exe
        
        # Run the compiled executable
        run_command = [binary_exe]
        try:
            process = subprocess.run(run_command, timeout=timeout, check=True)
            flag_exe = 1
        except:
            # print(func)
            if "process" in locals() and process:
                process.kill()
                process.wait()
            return flag_comp, flag_exe
    return flag_comp, flag_exe

def wrapper_func(args):
    # Unpack arguments and call the original function
    return execute_rate(*args)
def execute_rate_main(testsets, gen_results, num_workers=20, timeout=10, language='cpp', opt="-O0"):
    with multiprocessing.Pool(num_workers) as pool:
        tasks = [[testset["func_dep"], gen_result, testset["test"],\
                  timeout, language, opt]
            for testset, gen_result in zip(testsets, gen_results)
        ]
        eval_results = list(tqdm(pool.imap(wrapper_func, tasks), total=len(tasks)))

    comp, exe = zip(*eval_results)       
    return sum(exe) / len(exe)


if __name__ == "__main__":

    with open('/workspace/llm4binary/benchmark/data/humaneval-c-processed_20250402_2014.json','r') as f:
        json_new = json.load(f)

    test_dir = '/workspace/llm4binary/benchmark/text/llm4decompile-1.3b-v1.5-humaneval-c-vllm-ori'
    exe_all = []
    for i in [0,1,2,3]:
        gen_results = []
        for j in range(164):
            with open(os.path.join(test_dir, str(j*4+i)+'.c'), 'r') as f:
                gen_results.append(f.read().strip())
        eval_results, comp, exe = execute_rate_main(json_new, gen_results, num_workers=32, timeout=10, language='c', opt="-O0")

        exe_all.append(exe)
    for i in [0,1,2,3]:
        print(f'opt={i}, rate:{exe_all[i]/164.0}')


