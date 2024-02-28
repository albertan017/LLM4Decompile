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
from server.text_generation import TextGenerationServer, TextGenerationClient
import multiprocessing

logger.add(sys.stdout, colorize=False, format="{time} {level} {message}")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max_input_len", type=int, default=8192)
    parser.add_argument("--max_total_tokens", type=int, default=8800)
    parser.add_argument("--max_batch_prefill_tokens", type=int, default=72000)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--testset_path", type=str)
    parser.add_argument("--num_workers", type=int, default=16)
    return parser.parse_args()


def evaluate_func(params):
    c_func, c_test, c_func_decompile = (
        params["c_func"],
        params["c_test"],
        params["c_func_decompile"],
    )

    folder = "./"
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

    pid = os.getpid()
    # Define the C file and executable names
    c_file = os.path.join(folder, f"combine_{pid}.c")
    executable = os.path.join(folder, f"combine_{pid}")
    c_file_onlyfunc = os.path.join(folder, f"onlyfunc_{pid}.c")
    executable_onlyfunc = os.path.join(folder, f"onlyfunc_{pid}")

    if os.path.exists(executable):
        os.remove(executable)
    if os.path.exists(executable_onlyfunc):
        os.remove(executable_onlyfunc)

    with open(c_file, "w") as f:
        f.write(c_combine)
    with open(c_file_onlyfunc, "w") as f:
        f.write(c_onlyfunc)

    # Compile the C program to an assembly
    compile_command = f"gcc -S {c_file_onlyfunc} -o {executable_onlyfunc} -lm"
    try:
        subprocess.run(compile_command, shell=True, check=True)
        flag_compile = 1
    except:
        return flag_compile, flag_run
    finally:
        if os.path.exists(c_file_onlyfunc):
            os.remove(c_file_onlyfunc)
        if os.path.exists(executable_onlyfunc):
            os.remove(executable_onlyfunc)

    # Compile the C program to an executable
    compile_command = f"gcc {c_file} -o {executable} -lm"
    try:
        subprocess.run(compile_command, shell=True, check=True)
        flag_compile = 1
    except:
        return flag_compile, flag_run
    finally:
        if os.path.exists(c_file):
            os.remove(c_file)

    # Run the compiled executable
    run_command = f"{executable}"
    try:
        process = subprocess.run(
            run_command, shell=True, check=True, capture_output=True, timeout=5
        )
        flag_run = 1
    except subprocess.CalledProcessError as e:
        pass
    except Exception as e:
        pass
    finally:
        if os.path.exists(executable):
            os.remove(executable)

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
            "O0": "# This is the assembly code with O0 optimization:\n",
            "O1": "# This is the assembly code with O1 optimization:\n",
            "O2": "# This is the assembly code with O2 optimization:\n",
            "O3": "# This is the assembly code with O3 optimization:\n",
        }

        after = "\n# What is the source code?\n"

        inputs = []

        for testset in testsets:
            input_asm_prompt = testset["input_asm_prompt"]
            opt = testset["type"]
            prompt = opts[opt] + input_asm_prompt + after
            inputs.append(prompt)

        text_gen_server = TextGenerationServer(
            str(model_path),
            args.port,
            args.dtype,
            args.max_input_len,
            args.max_total_tokens,
            args.max_batch_prefill_tokens,
            args.num_shards,
        )

        text_gen_client = TextGenerationClient(
            port=args.port, stop_sequences=stop_sequences
        )

        gen_results_repeat = []
        logger.info(f"The exp will loop for {args.repeat} times....")
        for i in range(args.repeat):
            logger.info(f"The {i+1} loop...")
            loop = asyncio.get_event_loop()
            asyncio.set_event_loop(loop)
            gen_results = loop.run_until_complete(
                text_gen_client.generate_code_results(
                    inputs, args.max_new_tokens, num_outputs=1
                )
            )
            gen_results_repeat.append(gen_results)

    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return -1

    ret = decompile_pass_rate(testsets, gen_results_repeat, opts, args)
    return ret


def main():
    args = parse_args()
    ret = run_eval_pipeline(args)
    sys.exit(ret)


if __name__ == "__main__":
    main()
