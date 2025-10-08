import os
import sys
import json
import tempfile
import subprocess
from functools import partial
from tqdm import tqdm
import argparse
def process_one(sample_src, generator, solver):
    """
    Write sample_src to temp file (sample.c),
    run generator -> sample.cstr,
    run solver -> sample.h,
    read header, return header text.
    Any temp files are cleaned up automatically.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = os.path.join(tmpdir, "sample.c")
        output_path = os.path.join(tmpdir, "sample.cstr")
        header_path = os.path.join(tmpdir, "sample.h")

        # 1) dump the C鈥恠ource
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_src)

        try:
            # 2) run the generator
            subprocess.run(
                [generator, sample_path, "-o", output_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=1,
            )
            # 3) run the solver
            subprocess.run(
                ["stack", "exec", solver, "--", "-i", output_path, "-o", header_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=1,
            )

            # 4) read back the .h
            with open(header_path, "r", encoding="utf-8") as f:
                return f.read()

        # except subprocess.CalledProcessError as e:
        #     sys.stderr.write(
        #         f"[ERROR] sample failed:\n"
        #         f"  cmd: {e.cmd!r}\n"
        #         f"  returncode: {e.returncode}\n"
        #         f"  stdout: {e.stdout.decode(errors='ignore')}\n"
        #         f"  stderr: {e.stderr.decode(errors='ignore')}\n"
        #     )
        except Exception as e:
            return None


def main():
    p = argparse.ArgumentParser(description="Batch鈥恜rocess C samples into headers.")
    p.add_argument("--input_json", default="train_norm.json", help="Path to JSON file with a list of {{'code': 鈥} entries")
    p.add_argument("--output_name", default="train_type", help="Where to write the augmented JSON")
    p.add_argument("--generator", default="/psychec/psychecgen", help="Path to your generator executable")
    p.add_argument("--solver", default="/psychec/psychecsolver-exe", help="Name of your solver (for `stack exec 鈥)") 
    p.add_argument("--split", type=int, default=5, help="split the data to split parts")
    p.add_argument("--idx", type=int, default=0, help="index of the split")
    args = p.parse_args()

    # load
    with open(args.input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.split != 0:
        SPLIT = int(len(samples) / args.split)
        if args.idx == args.split - 1:
            samples = samples[SPLIT * args.idx:]
        else:
            samples = samples[SPLIT * args.idx:SPLIT * (args.idx + 1)]
    
    # pull out all the code鈥恠trings
    codes = [s["code_format"] for s in samples]############# code norm is the final expectation

    # prepare a partial that only needs the code
    worker = partial(process_one, generator=args.generator, solver=args.solver)

    memo = {}
    results = []
    count_non = 0
    for code in tqdm(codes):
        if code not in memo:
            header = worker(code)
            if header == None:
                count_non += 1
            memo[code] = header
        results.append(memo[code])
        if len(results) % 5000 == 0:
            print(f"len code:{len(codes)}, fail:{count_non}")

    for sample, header in zip(samples, results):
        sample["header"] = header

    # dump out
    with open(args.output_name+'_'+str(args.idx)+'.json', "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"len code:{len(codes)}, fail:{count_non}")

def folder():
    p = argparse.ArgumentParser(description="Batch鈥恜rocess C samples into headers.")
    p.add_argument("--input_folder", default="/workspace/llm4binary/type/evaluation/result/exebench-8800_github1000")
    # p.add_argument("--output_name", default="train_type", help="Where to write the augmented JSON")
    p.add_argument("--generator", default="../psychec/psychecgen", help="Path to your generator executable")
    p.add_argument("--solver", default="../psychec/psychecsolver-exe", help="Name of your solver (for `stack exec 鈥)") 
    # p.add_argument("--split", type=int, default=5, help="split the data to split parts")
    # p.add_argument("--idx", type=int, default=0, help="index of the split")
    args = p.parse_args()
    worker = partial(process_one, generator=args.generator, solver=args.solver)
    good = 0
    bad = 1
    for root, dirs, files in tqdm(os.walk(args.input_folder)):
        for filename in files:
            if filename.endswith(".c"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as f:
                    code = f.read()
                header = worker(code)
                with open(file_path.split('.c')[0] + ".h", 'w') as f:
                    if header:
                        good += 1
                        f.write(header)
                    else:
                        bad += 1
                        print(f'good:{good},bad:{bad}')
                        f.write("")


if __name__ == "__main__":
    # folder()
    main()



# ../psychec/psychecgen ./output/ori/3.c -o ./output/cstr/3.cstr
# stack exec ../psychec/psychecsolver-exe -- -i ./output/cstr/3.cstr -o ./output/header/3.h
