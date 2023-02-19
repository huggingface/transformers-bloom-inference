import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from functools import partial
from typing import Any, Callable, List, Tuple, Union

import torch
import torch.distributed as dist

from ..constants import DS_INFERENCE, DS_ZERO, HF_ACCELERATE, HF_CPU


# used for benchmarks
dummy_input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="model")
    group.add_argument(
        "--deployment_framework",
        type=str,
        choices=[HF_ACCELERATE, DS_INFERENCE, DS_ZERO, HF_CPU],
        default=HF_ACCELERATE,
    )
    group.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name to use",
    )
    group.add_argument(
        "--model_class",
        type=str,
        required=True,
        help="model class to use",
    )
    group.add_argument(
        "--dtype", type=str, required=True, choices=["bf16", "fp16", "int8", "fp32"], help="dtype for model"
    )
    group.add_argument(
        "--generate_kwargs",
        type=str,
        default='{"min_length": 100, "max_new_tokens": 100, "do_sample": false}',
        help="generate parameters. look at https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate to see the supported parameters",
    )
    group.add_argument("--max_input_length", type=int, help="max input length")
    group.add_argument("--max_batch_size", type=int, help="max supported batch size")

    return parser


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()

    args.dtype = get_torch_dtype(args.dtype)
    args.generate_kwargs = json.loads(args.generate_kwargs)

    return args


def run_rank_n(func: Callable, rank: int = 0, barrier: bool = False) -> None:
    # wrapper function for the rank to execute on
    def func_rank_n(*args, **kwargs):
        output = func(*args, **kwargs)
        if barrier:
            dist.barrier()
        return output

    # a dummy method that doesn't do anything
    def func_rank_other(*args, **kwargs):
        if barrier:
            dist.barrier()

    if dist.is_initialized():
        if dist.get_rank() == rank:
            return func_rank_n
        return func_rank_other
    else:
        return func


@run_rank_n
def print_rank_0(*args, **kwargs) -> None:
    print(*args, **kwargs)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "int8":
        return torch.int8
    elif dtype_str == "fp32":
        return torch.float32


def get_str_dtype(dtype_str: torch.dtype) -> str:
    if dtype_str == torch.bfloat16:
        return "bf16"
    elif dtype_str == torch.float16:
        return "fp16"
    elif dtype_str == torch.int8:
        return "int8"
    elif dtype_str == torch.float32:
        return "fp32"


def get_dummy_batch(batch_size: int, input_sentences: List[str] = None) -> List[str]:
    if input_sentences is None:
        input_sentences = copy.deepcopy(dummy_input_sentences)

    if batch_size > len(input_sentences):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    input_sentences = input_sentences[:batch_size]

    return input_sentences


def get_num_tokens_to_generate(max_new_tokens: int, allowed_max_new_tokens: int) -> int:
    if max_new_tokens is None:
        return allowed_max_new_tokens
    else:
        return min(max_new_tokens, allowed_max_new_tokens)


def run_and_log_time(execs: Union[List[partial], partial]) -> Tuple[Union[List[Any], Any], float]:
    # runs a function / list of functions and times them
    start_time = time.time()

    if type(execs) == list:
        results = []
        for f in execs:
            results.append(f())
    else:
        results = execs()

    time_elapsed = time.time() - start_time
    return results, time_elapsed


def pad_ids(arrays, padding, max_length=-1):
    # does left padding
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [[padding] * (max_length - len(array)) + array for array in arrays]

    return arrays


def get_exception_response(query_id: int, debug: bool = False):
    e_type, e_message, e_stack_trace = sys.exc_info()
    response = {"error": str(e_type.__name__), "message": str(e_message), "query_id": query_id}

    if debug:
        trace_back = traceback.extract_tb(e_stack_trace)

        # Format stacktrace
        stack_trace = []
        for trace in trace_back:
            stack_trace.append(
                "File : {}, Line : {}, Func.Name : {}, Message : {}".format(trace[0], trace[1], trace[2], trace[3])
            )

        response["stack_trace"] = stack_trace

    return response


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        cuda_visible_devices = get_cuda_visible_devices()
        if cuda_visible_devices is None:
            return 0
        return len(cuda_visible_devices)


def get_cuda_visible_devices() -> List[int]:
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        cuda_visible_devices = list(map(int, cuda_visible_devices.split(",")))
    return cuda_visible_devices
