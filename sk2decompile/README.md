# SK²Decompile

**SK²Decompile: LLM-based Two-Phase Binary Decompilation from Skeleton to Skin**

<p align="left">
    🚀&nbsp;<a href="#quick-start">Quick Start</a>
    | 🤖&nbsp;<a href="#training-pipeline">Training Pipeline</a>
    | 📊&nbsp;<a href="#evaluation">Evaluation</a>
    | 📝&nbsp;<a href="#citation">Citation</a>
</p>

## Overview

SK²Decompile is a novel two-phase framework for binary decompilation using Large Language Models (LLMs). Our approach decomposes the complex decompilation task into two manageable phases:

- **Phase 1 Structure Recovery (Skeleton)**: Transform binary/pseudo-code into obfuscated intermediate representations 🤗 [HF Link](https://huggingface.co/LLM4Binary/sk2decompile-struct-6.7b)
- **Phase 2 Identifier Naming (Skin)**: Generate human-readable source code with meaningful identifiers 🤗 [HF Link](https://huggingface.co/LLM4Binary/sk2decompile-ident-6.7)

This repository contains the complete implementation of our paper, including data preprocessing tools, training scripts, and evaluation benchmarks.

## 🏗️ Architecture

Our two-phase approach is inspired by the skeleton-to-skin metaphor:

```
Binary/Pseudo-code → [Phase 1: Skeleton] → Normalized IR → [Phase 2: Skin] → Source Code
                          ↓                                        ↓
                  (Structure Extraction)                   (Identifier Recovery)
```

## 📁 Repository Structure

```
SK2Decompile/
├── Preprocess/        # Data preprocessing and normalization tools
├── LLaMA-Factory/     # Supervised Fine-Tuning (SFT) implementation
├── verl/              # Reinforcement Learning (RL) with compiler-based rewards
├── evaluation/        # Comprehensive evaluation suite
└── README.md          # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.0+
- PyTorch 2.0+
- 40GB+ GPU memory (recommended)
- [Psychec](https://github.com/ltcmelo/psychec.git) (for data preprocessing)

### Installation

```bash
git clone https://github.com/yourusername/SK2Decompile.git
cd SK2Decompile
```

## 🤖 Training Pipeline

### Phase 0: Data Preprocessing

Transform raw pseudo-code into normalized representations suitable for training:

```bash
cd Preprocess

# Requirements
pip install tree-sitter==0.24.0 tree-sitter-c==0.23.4 tqdm

# Step 1: Normalize pseudo-code according to R2I standard
python3 normalize_pseudo.py --input_json exebench_c.json --output_json exebench_pseudonorm.json --key_name pseudo

# Step 2: Obfuscate source code to generate IR
python3 normalize_src_basedonpseudo.py --input_json exebench_pseudonorm.json --output_json exebench_norm_top0.json --top 0 --pseudo pseudo_norm

# Step 3: Format codes with clang-format
python3 format.py --input exebench_norm_top0.json --output exebench_format_top0.json

# Step 4: Infer types for obfuscated IR (used for compiler-based rewards)
python3 inf_type.py --input_json train_format_top0.json --output_name train_format_top0_type \
    --generator ../psychec/psychecgen --solver ../psychec/psychecsolver-exe --split 2 --idx 0
```

### Phase 1: Supervised Fine-Tuning (SFT)

Our two-phase SFT approach trains specialized models for each transformation:

#### Setup LLaMA-Factory
```bash
cd ../LLaMA-Factory
# Follow installation instructions in LLaMA-Factory/README.md
```

#### Train Models
```bash
# Train Skeleton Model (pseudo2norm)
llamafactory-cli train LLaMA-Factory/SK2DECOMPILE/train/pseudo2norm-example.yaml

# Train Skin Model (norm2code)
llamafactory-cli train LLaMA-Factory/SK2DECOMPILE/train/norm2code-example.yaml
```

**Sample Training Data:**
- Pseudo2Norm: `LLaMA-Factory/SK2DECOMPILE/data/pseudo2norm-examples.jsonl`
- Norm2Code: `LLaMA-Factory/SK2DECOMPILE/data/norm2code-examples.jsonl`

### Phase 2: Reinforcement Learning (RL)

Fine-tune models using compiler-based rewards for improved correctness:

#### Setup VERL
```bash
cd ../verl
# Follow installation instructions in verl/README.md
```

#### Run RL Training
```bash
bash verl/SK2DECOMPILE/train/sk2decompile-rl.sh
```

**RL Training Data:** `verl/SK2DECOMPILE/data/sk2decompile-rl-examples.parquet`

### Phase 3: Evaluation

**Inference**
```
from llm_server import llm_inference
from transformers import AutoTokenizer
import json
import argparse
import shutil
import os
from tqdm import tqdm

opts = ["O0", "O1", "O2", "O3"]
current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_path",type=str,default="LLM4Binary/llm4decompile-1.3b-v1.5")
    arg_parser.add_argument("--dataset_path",type=str,default='../data/exebench_test_normsrcpseudo_io.json')
    arg_parser.add_argument("--decompiler",type=str,default='asm')                    
    arg_parser.add_argument("--gpus", type=int, default=1)
    arg_parser.add_argument("--max_num_seqs", type=int, default=1)
    arg_parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    arg_parser.add_argument("--temperature", type=float, default=0)
    arg_parser.add_argument("--max_total_tokens", type=int, default=32768)
    arg_parser.add_argument("--max_new_tokens", type=int, default=4096)
    arg_parser.add_argument("--stop_sequences", type=str, default=None)
    arg_parser.add_argument("--recover_model_path", type=str, default=None, help="Path to the model to recover from, if any.")
    arg_parser.add_argument("--output_path", type=str, default='../result/exebench-1.3b-v2')
    arg_parser.add_argument("--only_save", type=int, default=0)
    arg_parser.add_argument("--strip", type=int, default=1)
    arg_parser.add_argument("--language", type=str, default='c')
    args = arg_parser.parse_args()

    before = "# This is the assembly code:\n"
    after = "\n# What is the source code?\n"

    if args.dataset_path.endswith('.json'):
        with open(args.dataset_path, "r") as f:
            print("===========")
            print(f"Loading dataset from {args.dataset_path}")
            print("===========")
            samples = json.load(f)
    elif args.dataset_path.endswith('.jsonl'):
        samples = []
        with open(args.dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

    if args.language == 'c':
        samples = [sample for sample in samples if sample['language'] == 'c']
    elif args.language == 'cpp':
        samples = [sample for sample in samples if sample['language'] == 'cpp']

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.stop_sequences is None:
        args.stop_sequences = [tokenizer.eos_token]


    filtered_samples = []
    for sample in tqdm(samples, desc="Filtering samples by token length"):
        if 'ida_strip_pseudo_norm' in sample:
            prompt = before + sample['ida_strip_pseudo_norm'] + after
            tokens = tokenizer.encode(prompt)
            if len(tokens) <= 12000:
                filtered_samples.append(sample)
            else:
                print(f"Discarded sample with {len(tokens)} tokens")
        else:
            filtered_samples.append(sample)
    
    samples = filtered_samples
    print(f"Filtered samples: {len(samples)} remaining")


    inputs = []
    infos = []
    for sample in samples:
        prompt = before + sample[args.decompiler].strip() + after
        sample['prompt_model1'] = prompt
        inputs.append(prompt)
        infos.append({
            "opt": sample["opt"],
            "language": sample["language"],
            "index": sample["index"],
            "func_name": sample["func_name"]
        })


    print("Starting first model inference...")
    gen_results = llm_inference(inputs, args.model_path,
                args.gpus,
                args.max_total_tokens,
                args.gpu_memory_utilization,
                args.temperature,
                args.max_new_tokens,
                args.stop_sequences)
    gen_results = [gen_result[0] for gen_result in gen_results]

    for idx in range(len(gen_results)):
        samples[idx]['gen_result_model1'] = gen_results[idx]

    inputs_recovery = []
    before_recovery = "# This is the normalized code:\n"
    after_recovery = "\n# What is the source code?\n"

    for idx, sample in enumerate(gen_results):
        prompt_recovery = before_recovery + sample.strip() + after_recovery
        samples[idx]['prompt_model2'] = prompt_recovery
        inputs_recovery.append(prompt_recovery)
    
    print("Starting recovery model inference...")
    gen_results_recovery = llm_inference(inputs_recovery, args.recover_model_path,
                args.gpus,
                args.max_total_tokens,
                args.gpu_memory_utilization,
                args.temperature,
                args.max_new_tokens,
                args.stop_sequences)
    gen_results_recovery = [gen_result[0] for gen_result in gen_results_recovery]


    for idx in range(len(gen_results_recovery)):
        samples[idx]['gen_result_model2'] = gen_results_recovery[idx]

    if args.output_path:
        if os.path.exists(args.output_path):
            shutil.rmtree(args.output_path)
        for opt in opts:
            os.makedirs(os.path.join(args.output_path, opt))

    if args.strip:
        print("Processing function name stripping...")
        for idx in range(len(gen_results_recovery)):
            one = gen_results_recovery[idx]
            func_name_in_gen = one.split('(')[0].split(' ')[-1].strip()
            if func_name_in_gen.strip() and func_name_in_gen[0:2] == '**':
                func_name_in_gen = func_name_in_gen[2:]
            elif func_name_in_gen.strip() and func_name_in_gen[0] == '*':
                func_name_in_gen = func_name_in_gen[1:]
            
            original_func_name = samples[idx]["func_name"]
            gen_results_recovery[idx] = one.replace(func_name_in_gen, original_func_name)
            samples[idx]["gen_result_model2_stripped"] = gen_results_recovery[idx]

    print("Saving inference results and logs...")
    for idx_sample, final_result in enumerate(gen_results_recovery):
        opt = infos[idx_sample]['opt']
        language = infos[idx_sample]['language']
        original_index = samples[idx_sample]['index']

        save_path = os.path.join(args.output_path, opt, f"{original_index}_{opt}.{language}")
        with open(save_path, "w") as f:
            f.write(final_result)

        log_path = save_path + ".log"
        log_data = {
            "index": original_index,
            "opt": opt,
            "language": language,
            "func_name": samples[idx_sample]["func_name"],
            "decompiler": args.decompiler,
            "input_asm": samples[idx_sample][args.decompiler].strip(),
            "prompt_model1": samples[idx_sample]['prompt_model1'],
            "gen_result_model1": samples[idx_sample]['gen_result_model1'],
            "prompt_model2": samples[idx_sample]['prompt_model2'],
            "gen_result_model2": samples[idx_sample]['gen_result_model2'],
            "final_result": final_result,
            "stripped": args.strip
        }
        
        if args.strip and "gen_result_model2_stripped" in samples[idx_sample]:
            log_data["gen_result_model2_stripped"] = samples[idx_sample]["gen_result_model2_stripped"]

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    json_path = os.path.join(args.output_path, 'inference_results.jsonl')
    with open(json_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    stats_path = os.path.join(args.output_path, 'inference_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Total samples processed: {len(samples)}\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Recovery model path: {args.recover_model_path}\n")
        f.write(f"Dataset path: {args.dataset_path}\n")
        f.write(f"Language: {args.language}\n")
        f.write(f"Decompiler: {args.decompiler}\n")
        f.write(f"Strip function names: {bool(args.strip)}\n")
        
        opt_counts = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}
        for sample in samples:
            opt_counts[sample['opt']] += 1
        
        f.write("\nSamples per optimization level:\n")
        for opt, count in opt_counts.items():
            f.write(f"  {opt}: {count}\n")

    print(f"Inference completed! Results saved to {args.output_path}")
    print(f"Total {len(samples)} samples processed.")
```

Comprehensive evaluation on standard benchmarks:

```bash
cd ../evaluation
# evaluate exe_rate
python evaluate_exe.py --json_file your_json_file_path
                       --dcompilers decompiler1,decompiler2,...,decompilerN
# evaluate r2i
python evaluate_r2i.py --json_file your_json_file_path
                       --dcompilers decompiler1,decompiler2,...,decompilerN
                       --output_path your_output_path
# evaluate gpt-judge
python gpt_judge.py --json_file your_json_file_path
                    --decompilers decompiler1,decompiler2,...,decompilerN
                    --opt OPT
                    --api_key your_openai_api_key
```

## 📊 Results

Our approach achieves state-of-the-art performance:

| Metric | Dataset | Improvement |
|--------|---------|-------------|
| **Re-executability** | HumanEval | **+21.6%** over GPT-5-mini |
| **R2I Score** | GitHub2025 | **+29.4%** over Idioms |

## 🔬 Key Innovations

1. **Two-Phase Decomposition**: Separating structure recovery from identifier prediction
2. **Compiler-Based RL**: Using compiler feedback as reward signal
3. **Generic Placeholders**: Language-agnostic intermediate representation
4. **Independent Optimization**: Separate RL objectives for each phase

## 📄 Citation

If you use SK²Decompile in your research, please cite our paper:

```bibtex
@article{sk2decompile2024,
  title={SK²Decompile: From Skeleton to Skin - A Two-Phase Approach for Binary Decompilation},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Support for additional architectures (ARM, RISC-V)
- Integration with more decompilation tools
- Improved intermediate representations
- Multi-language support

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

We thank the developers of:
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the SFT framework
- [VERL](https://github.com/volcengine/verl) for the RL implementation
- [Psychec](https://github.com/ltcmelo/psychec.git) for C type inference

---

For detailed documentation on each component, please refer to the individual README files in each directory.
