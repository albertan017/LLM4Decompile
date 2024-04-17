import subprocess
import asyncio
from transformers import AutoTokenizer
import os
import json
from loguru import logger
import traceback
from argparse import ArgumentParser
from pathlib import Path
import sys
from tqdm import tqdm
import multiprocessing
import tempfile
from vllm import LLM, SamplingParams

logger.add(sys.stdout, colorize=False, format="{time} {level} {message}")
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.82)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_total_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--testset_path", type=str)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    return parser.parse_args()


def evaluate_func(params):
    c_func, c_test, c_func_decompile = (
        params["c_func"],
        params["c_test"],
        params["c_func_decompile"],
    )

    timeout = 10
    flag_compile = 0
    flag_run = 0
    c_include = ""
    for line in c_func.split("\n"):
        if "#include" in line:
            c_include += line + "\n"
            c_func = c_func.replace(line, "")
    for line in c_test.split("\n"):
        if "#include" in line:
            c_include += line + "\n"
            c_test = c_test.replace(line, "")
    c_combine = c_include + "\n" + c_func_decompile + "\n" + c_test
    c_onlyfunc = c_include + "\n" + c_func_decompile

    with tempfile.TemporaryDirectory() as temp_dir:
        pid = os.getpid()
        c_file = os.path.join(temp_dir, f"combine_{pid}.c")
        executable = os.path.join(temp_dir, f"combine_{pid}")
        c_file_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}.c")
        executable_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}")

        with open(c_file, "w") as f:
            f.write(c_combine)
        with open(c_file_onlyfunc, "w") as f:
            f.write(c_onlyfunc)

        # Compile the C program to an assembly
        compile_command = [
            "gcc",
            "-S",
            c_file_onlyfunc,
            "-o",
            executable_onlyfunc,
            "-lm",
        ]
        try:
            subprocess.run(compile_command, check=True, timeout=timeout)
            flag_compile = 1
        except:
            return flag_compile, flag_run

        # Compile the C program to an executable
        compile_command = ["gcc", c_file, "-o", executable, "-lm"]
        try:
            subprocess.run(compile_command, check=True, timeout=timeout)
            flag_compile = 1
        except:
            return flag_compile, flag_run

        # Run the compiled executable
        run_command = [executable]
        try:
            process = subprocess.run(
                run_command, capture_output=True, text=True, timeout=timeout, check=True
            )
            flag_run = 1
        except:
            if "process" in locals() and process:
                process.kill()
                process.wait()
            return flag_compile, flag_run

    return flag_compile, flag_run


def decompile_pass_rate(testsets, gen_results_repeat, opts, args):
    all_stats = []

    for gen_index, gen_results in enumerate(gen_results_repeat):
        with multiprocessing.Pool(args.num_workers) as pool:
            tasks = [
                {
                    "c_func": testset["c_func"],
                    "c_test": testset["c_test"],
                    "c_func_decompile": output[0],
                }
                for testset, output in zip(testsets, gen_results)
            ]

            eval_results = list(tqdm(pool.imap(evaluate_func, tasks), total=len(tasks)))

        pool.terminate()
        pool.join()

        stats = {opt: {"compile": 0, "run": 0, "total": 0} for opt in opts}
        for idx, (testset, output, flag) in enumerate(
            tqdm(
                zip(testsets, gen_results, eval_results),
                total=len(testsets),
                desc="Evaluating",
            )
        ):
            c_func_decompile = output[0]
            c_func = testset["c_func"]
            c_test = testset["c_test"]

            flag_compile, flag_run = flag[0], flag[1]
            opt = testset["type"]

            stats[opt]["total"] += 1
            if flag_compile:
                stats[opt]["compile"] += 1
            if flag_run:
                stats[opt]["run"] += 1

        all_stats.append(stats)

    # average
    avg_stats = {opt: {"compile": 0, "run": 0, "total": 0} for opt in opts}
    for stats in all_stats:
        for opt in opts:
            avg_stats[opt]["compile"] += stats[opt]["compile"]
            avg_stats[opt]["run"] += stats[opt]["run"]
            avg_stats[opt]["total"] += stats[opt]["total"]

    for opt in opts:
        avg_stats[opt]["compile"] /= len(gen_results_repeat)
        avg_stats[opt]["run"] /= len(gen_results_repeat)
        avg_stats[opt]["total"] /= len(gen_results_repeat)

    for opt, data in avg_stats.items():
        compile_rate = data["compile"] / data["total"] if data["total"] > 0 else 0
        run_rate = data["run"] / data["total"] if data["total"] > 0 else 0
        print(
            f"Optimization {opt}: Compile Rate: {compile_rate:.4f}, Run Rate: {run_rate:.4f}"
        )

    return 0


def run_eval_pipeline(args: ArgumentParser) -> int:
    model_path = Path(args.model_path)
    if not model_path.exists() or not model_path.is_dir():
        logger.error(f"Invalid model {model_path}")
        return -1

    try:
        testsets = json.load(open(args.testset_path, "r"))
        logger.info(f"Loaded testset with {len(testsets)} cases")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        stop_sequences = [tokenizer.eos_token]

        opts = {
            "O0": "# This is the assembly code:\n",
            "O1": "# This is the assembly code:\n",
            "O2": "# This is the assembly code:\n",
            "O3": "# This is the assembly code:\n",
        }

        after = "\n# What is the source code?\n"
        inputs = []
        for testset in testsets:
            input_asm_prompt = testset["input_asm_prompt"]
            opt = testset["type"]
            prompt = opts[opt] + input_asm_prompt + after
            inputs.append(prompt)

        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.gpus,
            max_model_len=args.max_total_tokens,
            # max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            stop=stop_sequences,
        )

        gen_results_repeat = []
        logger.info(f"The exp will loop for {args.repeat} times....")
        for i in range(args.repeat):
            logger.info(f"The {i+1} loop...")
            gen_results = llm.generate(inputs, sampling_params)
            gen_results = [[output.outputs[0].text] for output in gen_results]
            gen_results_repeat.append(gen_results)

    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return -1

    save_data = []
    for testset, res in zip(testsets, gen_results_repeat[0]):
        testset["output"] = res[0]
        save_data.append(testset)

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=True)

    ret = decompile_pass_rate(testsets, gen_results_repeat, opts, args)
    return ret


def main():
    args = parse_args()
    ret = run_eval_pipeline(args)
    sys.exit(ret)


if __name__ == "__main__":
    main()
