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
    arg_parser.add_argument("--model_path",type=str,default="LLM4Binary/sk2decompile-struct-6.7b")
    arg_parser.add_argument("--dataset_path",type=str,default='reverse_sample.json')
    arg_parser.add_argument("--decompiler",type=str,default='ida_pseudo_norm')                    
    arg_parser.add_argument("--gpus", type=int, default=1)
    arg_parser.add_argument("--max_num_seqs", type=int, default=1)
    arg_parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    arg_parser.add_argument("--temperature", type=float, default=0)
    arg_parser.add_argument("--max_total_tokens", type=int, default=32768)
    arg_parser.add_argument("--max_new_tokens", type=int, default=4096)
    arg_parser.add_argument("--stop_sequences", type=str, default=None)
    arg_parser.add_argument("--recover_model_path", type=str, default='LLM4Binary/sk2decompile-ident-6.7', help="Path to the model to recover from, if any.")
    arg_parser.add_argument("--output_path", type=str, default='./result/sk2decompile')
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


    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.stop_sequences is None:
        args.stop_sequences = [tokenizer.eos_token]

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
