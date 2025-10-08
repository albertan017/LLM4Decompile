import re
import os
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
import argparse

def good_func(func):
    func = '{'.join(func.split('{')[1:])
    func_sp = func.split('\n')
    total = 0
    for line in func_sp:
        if len(line.strip())>=3:
            total+=1
    if total>3 and total<300:
        return True
    return False

def strip_empty(code):
    return "\n".join(line for line in code.splitlines() if line.strip())
def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

import subprocess

def format_with_clang(func: str, style: str = "Google") -> str:
    # Build the command
    if not func:
        return None
    cmd = ["clang-format", f"--style={style}"]
    try:
        proc = subprocess.run(
            cmd,
            input=func,
            text=True,
            capture_output=True,
            check=True,
            timeout=0.5
        )
        return proc.stdout
    except:
        # print("clang-format failed")
        return None


def process_record(record):
    src = record.get("code_norm", "")
    no_comments = comment_remover(src)
    formatted = format_with_clang(no_comments)
    if formatted is None:
        return {}
    cleaned = strip_empty(formatted)
    record["code_format"] = cleaned
    if not good_func(cleaned):
        return {}
    return record

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel clang-format + strip for JSONL data"
    )
    parser.add_argument("--input_json", type=str, default="train_norm.json")
    parser.add_argument("--output_json", type=str, default="train_format.json")
    parser.add_argument(
        "-j", "--jobs", type=int, default=cpu_count())
    args = parser.parse_args()

    # 1) Load data
    with open(args.input_json, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    # 2) Process in parallel with a progress bar
    with Pool(processes=args.jobs) as pool:
        results = list(tqdm(pool.imap(process_record, data),
                            total=len(data),
                            desc="Processing format"))

    results = [record for record in results if record]###############only keep good functions that have 3 lines
    # 3) Write out
    with open(args.output_json, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4)
    
    import random
    data_sample = random.sample(results, 2)
    for record in data_sample:
        print('________________format ori___________________')
        print(record['code'])
        print('________________format format___________________')
        print(record['code_format'])
