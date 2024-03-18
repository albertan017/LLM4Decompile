import glob
import json
import subprocess
import os
import multiprocessing
import re
import argparse

zeros_pattern = r"^0+\s"  # 0000000000000...
OPT = ["O0", "O1", "O2", "O3"]  # Optimization states


def compile_and_write(input_file, output_file):
    base_output_file = input_file.replace(".c", "")
    asm_all = {}
    input_text = open(input_file).read()  # Read input file
    if "/* Variables and functions */" in input_text:
        # Exclude macro and types
        input_text = input_text.split("/* Variables and functions */")[-1]
        input_text = "\n\n".join(input_text.split("\n\n")[1:])  # Exclude variables
        ##### begin of remove __attribute__
        input_text = input_text.replace("__attribute__((used)) ", "")
        ##### end of remove __attribute__
    try:
        for opt_state in OPT:
            obj_output = base_output_file + "_" + opt_state + ".o"
            asm_output = base_output_file + "_" + opt_state + ".s"

            # Compile the C program to object file
            subprocess.run(
                ["gcc", "-c", "-o", obj_output, input_file, "-" + opt_state],
                check=True,
            )

            # Generate assembly code from object file using objdump
            subprocess.run(
                f"objdump -d {obj_output} > {asm_output}",
                shell=True,  # Use shell to handle redirection
                check=True,
            )

            with open(asm_output) as f:
                asm = f.read()
                ##### start of clean up
                asm_clean = ""
                asm = asm.split("Disassembly of section .text:")[-1].strip()
                for tmp in asm.split("\n"):
                    tmp_asm = tmp.split("\t")[-1]  # remove the binary code
                    tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments
                    asm_clean += tmp_asm + "\n"
                if len(asm_clean.split("\n")) < 4:
                    raise ValueError("compile fails")
                asm = asm_clean
                ##### end of clean up

                ##### start of filter digits and attribute
                asm = re.sub(zeros_pattern, "", asm)
                asm = asm.replace("__attribute__((used)) ", "")
                ##### end of filter digits

                asm_all["opt-state-" + opt_state] = asm

            # Remove the object file
            if os.path.exists(obj_output):
                os.remove(obj_output)

    except Exception as e:
        print(f"Error in file {input_file}: {e}")
        return
    finally:
        # Remove the assembly output files
        for opt_state in OPT:
            asm_output = base_output_file + "_" + opt_state + ".s"
            if os.path.exists(asm_output):
                os.remove(asm_output)

    sample = {
        "name": input_file,
        "input": input_text,  # Use the processed input text
        "input_ori": open(input_file).read(),
        "output": asm_all,  # Use the asm_all
    }

    # Write to file
    write_to_file(output_file, sample)


def write_to_file(file_path, data):
    with multiprocessing.Lock():
        with open(file_path, "a") as f:
            json.dump(data, f)
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile C files and generate JSONL output."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory where AnghaBench files are located.",
    )
    parser.add_argument("--output", required=True, help="Path to JSONL output file.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root = args.root
    jsonl_output_file = args.output
    files = glob.glob(f"{root}/**/*.c", recursive=True)

    with multiprocessing.Pool(32) as pool:
        from functools import partial

        compile_write_func = partial(compile_and_write, output_file=jsonl_output_file)
        pool.map(compile_write_func, files)


if __name__ == "__main__":
    main()
