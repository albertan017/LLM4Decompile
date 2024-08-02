# LLM4Decompile-Ref

<p align="left">
    📊&nbsp;<a href="#evaluation">Results</a>
    | 🤗&nbsp;<a href="#models">Models</a>
    | 🚀&nbsp;<a href="#quick-start">Quick Start</a>
    | 📚&nbsp;<a href="#humaneval-decompile">HumanEval-Decompile</a>
    | 📝&nbsp;<a href="https://arxiv.org/abs/2403.05286">Paper</a>
</p>

## Updates
* [2024-06-19]: Release [V2](https://huggingface.co/LLM4Binary/llm4decompile-6.7b-v2) series (LLM4Decompile-Ref). V2 (1.3B-22B), building upon **Ghidra**, are trained on 2 billion tokens to **refine** the decompiled pseudo-code from Ghidra. The 22B-V2 version outperforms the 6.7B-V1.5 by an additional 40.1%.

## About
* **LLM4Decompile-Ref** refines the pseudo-code decompiled by Ghidra.


## Evaluation

### Framework
<p align="center">
<img src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/decompile-refine.png" alt="image" width="400" height="auto">
</p>

This approach differs from that in LLM4Decompile-End only in terms of the LLM's input, which in the case of Refined-Decompile comes from **Ghidra**'s decompilation output. Specifically, Ghidra is used to decompile the binary, and then the LLM is fine-tuned to enhance Ghidra's output. While Ghidra produces high-level pseudo-code that may suffer from readability issues and syntax errors, it effectively preserves the underlying logic. Refining this pseudo-code significantly mitigates the challenges associated with understanding the obscure ASM.

### Results

<p align="center">
<img src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/results_refine.png" alt="results" width="800" height="auto">
</p>

## Models
Our LLM4Decompile includes models with sizes between 1.3 billion and 33 billion parameters, and we have made these models available on Hugging Face.

| Model                 | Checkpoint                                                        | Size | Re-executability       | Note |
|-----------------------|-------------------------------------------------------------------|------|---------------------|----------------------|
| **llm4decompile-1.3b-v1.5**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v1.5)   | 1.3B | 27.3%   | Note 3 |
| **llm4decompile-6.7b-v1.5**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-6.7b-v1.5)   | 6.7B | 45.4%   | Note 3 |
| **llm4decompile-1.3b-v2**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v2)   | 1.3B | **46.0%**   | Note 4 |
| **llm4decompile-6.7b-v2**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-6.7b-v2)   | 6.7B | **52.7%**   | Note 4 |
| **llm4decompile-22b-v2**| 🤗 [HF Link](https://huggingface.co/LLM4Binary/llm4decompile-22b-v2)   | 22B | **63.6%**   | Note 4 |

Note 3: V1.5 series are trained with a larger dataset (15B tokens) and a maximum token size of 4,096, with remarkable performance (over 100% improvement) compared to the previous model.

Note 4: V2 series are built upon **Ghidra** and trained on 2 billion tokens to **refine** the decompiled pseudo-code from Ghidra.

## Quick Start
Here is an example of how to use our model (Only for V2. For previous models, please check the corresponding model page at HF).

1. Install Ghidra
Download [Ghidra](https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.0.3_build/ghidra_11.0.3_PUBLIC_20240410.zip) to the current folder. You can also check the [page](https://github.com/NationalSecurityAgency/ghidra/releases) for other versions. Unzip the package to the current folder.
In bash, you can use the following:
```bash
cd LLM4Decompile/ghidra
wget https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.0.3_build/ghidra_11.0.3_PUBLIC_20240410.zip
unzip ghidra_11.0.3_PUBLIC_20240410.zip
```
2. Install Java-SDK-17
Ghidra 11 is dependent on Java-SDK-17, a simple way to install the SDK on Ubuntu:
```bash
apt-get update
apt-get upgrade
apt install openjdk-17-jdk openjdk-17-jre
```
Please check [Ghidra install guide](https://htmlpreview.github.io/?https://github.com/NationalSecurityAgency/ghidra/blob/Ghidra_11.1.1_build/GhidraDocs/InstallationGuide.html) for other platforms. 

3. Use Ghidra Headless to decompile binary (demo.py)

Note: **Replace** func0 with the function name you want to decompile.

**Preprocessing:** Compile the C code into binary, and disassemble the binary into assembly instructions.
```python
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
```

Ghidra pseudo-code may look like this:
```c
undefined4 func0(float param_1,long param_2,int param_3)
{
  int local_28;
  int local_24;
  
  local_24 = 0;
  do {
    local_28 = local_24;
    if (param_3 <= local_24) {
      return 0;
    }
    while (local_28 = local_28 + 1, local_28 < param_3) {
      if ((double)((ulong)(double)(*(float *)(param_2 + (long)local_24 * 4) -
                                  *(float *)(param_2 + (long)local_28 * 4)) &
                  SUB168(_DAT_00402010,0)) < (double)param_1) {
        return 1;
      }
    }
    local_24 = local_24 + 1;
  } while( true );
}
```
4. Refine pseudo-code using LLM4Decompile (demo.py)

**Decompilation:** Use LLM4Decompile-Ref to refine the Ghidra pseudo-code into C:
```python
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

```

## HumanEval-Decompile
Data for the pseudo-code are stored in ``llm4decompile/decompile-eval/decompile-eval-executable-gcc-ghidra.json``, using JSON list format. There are 164*4 (O0, O1, O2, O3) samples, each with five keys:

*   ``task_id``: indicates the ID of the problem.
*   ``type``: the optimization stage, is one of [O0, O1, O2, O3].
*   ``c_func``: C solution for HumanEval problem. 
*   ``c_test``: C test assertions.
*   ``input_asm_prompt``: Ghidra decompiled result.

Please check the [evaluation scripts](https://github.com/albertan017/LLM4Decompile/tree/main/evaluation).

## Thanks
The Ghidra Headless script is originated from [galoget](https://github.com/galoget/ghidra-headless-scripts)
