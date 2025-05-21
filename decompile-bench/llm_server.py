from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import os
import json
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "true"

inputs = []
def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_total_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--stop_sequences", type=str, default=None)
    parser.add_argument("--testset_path", type=str)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()

# def llm_inference(inputs, args):
#     llm = LLM(
#         model=args.model_path,
#         tensor_parallel_size=args.gpus,
#         max_model_len=args.max_total_tokens,
#         gpu_memory_utilization=args.gpu_memory_utilization,
#     )

#     sampling_params = SamplingParams(
#         temperature=args.temperature,
#         max_tokens=args.max_new_tokens,
#         stop=args.stop_sequences,
#     )

#     gen_results = llm.generate(inputs, sampling_params)
#     gen_results = [[output.outputs[0].text] for output in gen_results]

#     return gen_results


def llm_inference(inputs,
                  model_path,
                  gpus=1,
                  max_total_tokens=8192,
                  gpu_memory_utilization=0.95,
                  temperature=0,
                  max_new_tokens=512,
                  stop_sequences=None):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=gpus,
        max_model_len=max_total_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop=stop_sequences,
    )

    gen_results = llm.generate(inputs, sampling_params)
    gen_results = [[output.outputs[0].text] for output in gen_results]

    return gen_results

if __name__ == "__main__":
    args = parse_args()
    with open(args.testset_path, "r") as f:
        samples = json.load(f)
        before = "# This is the assembly code:\n"
        after = "\n# What is the source code?\n"
        for sample in samples:
            prompt = before + sample["input_asm_prompt"].strip() + after
            inputs.append(prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.stop_sequences is None:
        args.stop_sequences = [tokenizer.eos_token]
    gen_results = llm_inference(inputs, args.model_path,
                  args.gpus,
                  args.max_total_tokens,
                  args.gpu_memory_utilization,
                  args.temperature,
                  args.max_new_tokens,
                  args.stop_sequences)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    idx = 0
    for gen_result in gen_results:
        with open(args.output_path + '/' + str(idx) + '.c', 'w') as f:
            f.write(gen_result[0])
        idx += 1