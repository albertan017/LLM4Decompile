from llm_server import llm_inference
from metrics.cal_execute_rate import execute_rate_main
from transformers import AutoTokenizer
import json
import argparse
import shutil
import os
opts = ["O0", "O1", "O2", "O3"]

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_path",type=str,default="LLM4Binary/llm4decompile-1.3b-v1.6")
    arg_parser.add_argument("--dataset_path",type=str,default='./data/humaneval-decompile.json')
    arg_parser.add_argument("--decompiler",type=str,default='asm')                    
    arg_parser.add_argument("--gpus", type=int, default=1)
    arg_parser.add_argument("--max_num_seqs", type=int, default=1)
    arg_parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    arg_parser.add_argument("--temperature", type=float, default=0)
    arg_parser.add_argument("--max_total_tokens", type=int, default=30000)
    arg_parser.add_argument("--max_new_tokens", type=int, default=512)
    arg_parser.add_argument("--stop_sequences", type=str, default=None)
    arg_parser.add_argument("--output_path", type=str, default='./data/humaneval')
    arg_parser.add_argument("--only_save", type=int, default=0)
    args = arg_parser.parse_args()

    before = "# This is the assembly code:\n"
    after = "\n# What is the source code?\n"
    with open(args.dataset_path, "r") as f:
        samples = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.stop_sequences is None:
        args.stop_sequences = [tokenizer.eos_token]

    results = []
    inputs = []
    infos = []
    for sample in samples:
        prompt = before + \
                sample[args.decompiler].strip() + \
                after
        inputs.append(prompt)
        infos.append({
            "opt": sample["opt"],
            "language": sample["language"]
        })
    # llm generation
    gen_results = llm_inference(inputs, args.model_path,
                args.gpus,
                args.max_total_tokens,
                args.gpu_memory_utilization,
                args.temperature,
                args.max_new_tokens,
                args.stop_sequences)
    gen_results = [gen_result[0] for gen_result in gen_results]
    
    gen_results_opt = {}
    if args.output_path:
        if os.path.exists(args.output_path):
            shutil.rmtree(args.output_path)
        for opt in opts:
            os.makedirs(os.path.join(args.output_path, opt))
            gen_results_opt[opt] = []
        for idx_sample, sample in enumerate(gen_results):
            save_path = os.path.join(args.output_path, infos[idx_sample]['opt'], \
                        f"{idx_sample}_{infos[idx_sample]['opt']}.{infos[idx_sample]['language']}")
            with open(save_path, "w") as f:
                f.write(gen_results[idx_sample])
            gen_results_opt[infos[idx_sample]['opt']].append(gen_results[idx_sample])
    # executable rate
    if not args.only_save:
        eval_results, comp, exe = execute_rate_main(samples, gen_results, num_workers=32)
        results = {"O0":0, "O1":0, "O2":0, "O3":0}
        total = {"O0":0, "O1":0, "O2":0, "O3":0}
        for idx,res in enumerate(eval_results):
            results[infos[idx]['opt']] += res[1]
            total[infos[idx]['opt']] += 1
        name = args.dataset_path.split('/')[-1]
        print(f'dataset: {name}')
        for opt in opts:
            exe = results[opt]*1.0/total[opt]
            print(f'{opt}: {exe*100:.2f}')
            with open(os.path.join(args.output_path, args.output_path.split('/')[-1]+'_results.txt'), 'a') as f:
                f.write(f'{opt}: {exe*100:.2f}\n')
            