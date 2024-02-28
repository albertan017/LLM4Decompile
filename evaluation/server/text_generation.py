import asyncio
import numpy as np
import os
import random
import socket
import subprocess
import time

from loguru import logger
from text_generation import AsyncClient
from tqdm import tqdm
from typing import List


class TextGenerationServer:
    def __init__(
        self,
        model_id: str,
        port: int,
        dtype: str,
        max_input_len: int,
        max_total_tokens: int,
        max_batch_prefill_tokens: int,
        num_shards: int,
    ):

        master_port = random.randint(10_000, 20_000)
        args = [
            "text-generation-launcher",
            "--model-id",
            model_id,
            "--port",
            str(port),
            "--master-port",
            str(master_port),
        ]

        args.extend(["--num-shard", str(num_shards)])
        args.extend(["--dtype", dtype])
        args.extend(["--max-input-length", str(max_input_len)])
        args.extend(["--max-total-tokens", str(max_total_tokens)])
        args.extend(["--max-batch-prefill-tokens", str(max_batch_prefill_tokens)])

        logger.info(" ".join(args))
        self.launcher = subprocess.Popen(args, stdout=subprocess.DEVNULL)
        logger.info("Waiting for text generation server to start...")

        def webserver_ready():
            try:
                socket.create_connection(("127.0.0.1", 8080), timeout=1).close()
                return True
            except (socket.timeout, ConnectionRefusedError):
                return False

        while not webserver_ready():
            time.sleep(10)
        logger.info("Text generation webserver ready")

    def __del__(self):
        self.launcher.terminate()
        self.launcher.wait()


class TextGenerationClient:
    def __init__(self, port, stop_sequences: List[str]):
        self.client = AsyncClient(f"http://127.0.0.1:{port}", timeout=9999)
        self.stop_sequences = stop_sequences

    async def generate(
        self,
        input: str,
        max_new_tokens: int,
        do_sample: bool,
        pbar: tqdm,
        **kwargs,
    ) -> str:
        try:
            if do_sample:
                top_p = kwargs.get("top_p", 0.95)
                temperature = kwargs.get("temperature", 0.8)
                output = await self.client.generate(
                    input,
                    max_new_tokens=max_new_tokens,
                    stop_sequences=self.stop_sequences,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                output = await self.client.generate(
                    input,
                    max_new_tokens=max_new_tokens,
                    stop_sequences=self.stop_sequences,
                    do_sample=do_sample,
                )
            generated_text = output.generated_text
            for stop_sequence in self.stop_sequences:
                generated_text = generated_text.replace(stop_sequence, "")

        except Exception as e:
            generated_text = ""
            logger.error(e)
        pbar.update()
        return generated_text

    async def generate_code_results(
        self,
        inputs: List[str],
        max_new_tokens: int,
        num_outputs: int,
        task_size: int = 50,
        **kwargs,
    ) -> np.array:
        with tqdm(
            total=len(inputs * num_outputs), desc="Fetching code generation results"
        ) as pbar:
            results = []
            max_new_tokens = max_new_tokens if max_new_tokens > 0 else 32
            do_sample = num_outputs > 1
            requests = [input for input in inputs for _ in range(num_outputs)]
            for i in range(0, len(requests), task_size):
                tasks = []
                for input in requests[i : i + task_size]:
                    task = asyncio.ensure_future(
                        self.generate(input, max_new_tokens, do_sample, pbar, **kwargs)
                    )
                    tasks.append(task)
                for result in await asyncio.gather(*tasks):
                    results.append(result)
            results = np.array(results).reshape(len(inputs), num_outputs)
        return results
