import argparse
import gc
from functools import partial

import torch

from .constants import DS_INFERENCE, DS_ZERO
from .model_handler.deployment import ModelDeployment
from .models import start_inference_engine
from .utils import (
    GenerateRequest,
    create_generate_request,
    get_argument_parser,
    get_dummy_batch,
    get_world_size,
    parse_args,
    print_rank_0,
    run_and_log_time,
)


def benchmark_generation(model: ModelDeployment, request: GenerateRequest, cycles: int = 5):
    # run benchmarks for number of cycles
    total_new_tokens_generated = 0
    for _ in range(cycles):
        response = model.generate(request=request)
        total_new_tokens_generated += sum(new_tokens for new_tokens in response.num_generated_tokens)
    return total_new_tokens_generated


def get_benchmark_results(
    benchmark_time: float, initialization_time: float, total_new_tokens_generated: int, batch_size: int, cycles: int
) -> str:
    throughput = total_new_tokens_generated / benchmark_time
    latency = benchmark_time / cycles
    return f"""
*** Performance stats:
Throughput (including tokenization) = {throughput:.2f} tokens/sec
Throughput (including tokenization) = {1000 / throughput:.2f} msecs/token
Model loading time = {initialization_time:.2f} secs
Total tokens generated = {total_new_tokens_generated} with batch size = {batch_size}
Latency = {latency:.2f} secs
Model loading time + generation time per batch = {initialization_time + latency:.2f} secs
"""


def benchmark_end_to_end(args: argparse.Namespace) -> None:
    model, initialization_time = run_and_log_time(partial(ModelDeployment, args=args, grpc_allowed=False))

    request = create_generate_request(get_dummy_batch(args.batch_size), args.generate_kwargs)

    print_rank_0(f"generate_kwargs = {args.generate_kwargs}")
    print_rank_0(f"batch_size = {args.batch_size}")

    # warmup is a must if measuring speed as it's when all the optimizations are performed
    # e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
    response = model.generate(request=request)

    for i, (o, _) in zip(request.text, zip(response.text, response.num_generated_tokens)):
        print_rank_0(f"{'-' * 60}\nin = {i}\nout = {o}\n")

    if args.benchmark_cycles > 0:
        print_rank_0("*** Running benchmark")

        torch.cuda.empty_cache()
        gc.collect()

        # warm up
        model.generate(request=request)
        torch.cuda.synchronize()

        # benchmark
        total_new_tokens_generated, benchmark_time = run_and_log_time(
            partial(benchmark_generation, model=model, request=request, cycles=args.benchmark_cycles)
        )

        # with ZeRO every GPU is generating batch_size * sequence_length tokens
        if args.deployment_framework == DS_ZERO:
            total_new_tokens_generated *= get_world_size()

        print_rank_0(
            get_benchmark_results(
                benchmark_time, initialization_time, total_new_tokens_generated, args.batch_size, args.benchmark_cycles
            )
        )


def get_args() -> argparse.Namespace:
    parser = get_argument_parser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--benchmark_cycles", type=int, default=0, help="additionally run benchmark")
    group.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    group.add_argument("--batch_size", default=1, type=int, help="batch size")
    group.add_argument("--cpu_offload", action="store_true", help="whether to activate CPU offload for DS ZeRO")

    args = parse_args(parser)

    launched_with_deepspeed = args.deployment_framework in [DS_INFERENCE, DS_ZERO]

    assert args.max_batch_size == None, "max_batch_size is not supported with benchmark"

    if not launched_with_deepspeed:
        assert args.local_rank == None, "local_rank must be None if not launched with DeepSpeed"

    if args.cpu_offload:
        assert args.deployment_framework == DS_ZERO, "cpu_offload only works with DS_ZeRO"

    return args


def main() -> None:
    args = get_args()
    start_inference_engine(args.deployment_framework)
    benchmark_end_to_end(args)


if __name__ == "__main__":
    main()
