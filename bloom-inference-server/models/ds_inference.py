import argparse
import glob
import io
import json
import os
from argparse import Namespace
from functools import partial

import torch
import torch.distributed as dist

import deepspeed
import mii
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils import GenerateRequest, GenerateResponse, get_filter_dict, get_str_dtype, print_rank_n, run_rank_n

from .model import Model, check_max_input_length, get_downloaded_model_path, get_stopping_criteria


# basic DeepSpeed inference model class for benchmarking
class DSInferenceModel(Model):
    def __init__(self, args: Namespace) -> None:
        print_rank_n("Loading model...")
        world_size = int(os.getenv("WORLD_SIZE", "1"))

        downloaded_model_path = get_downloaded_model_path(args.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(downloaded_model_path)
        self.pad = self.tokenizer.pad_token_id

        # create dummy tensors for allocating space which will be filled with
        # the actual weights while calling deepspeed.init_inference in the
        # following code
        with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
            self.model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(downloaded_model_path), torch_dtype=torch.bfloat16
            )
        self.model = self.model.eval()

        if args.dtype in [torch.float16, torch.int8]:
            # We currently support the weights provided by microsoft (which are
            # pre-sharded)
            if args.use_pre_sharded_checkpoints:
                checkpoints_json = os.path.join(downloaded_model_path, "ds_inference_config.json")

                self.model = deepspeed.init_inference(
                    self.model,
                    mp_size=world_size,
                    base_dir=downloaded_model_path,
                    dtype=args.dtype,
                    checkpoint=checkpoints_json,
                    replace_with_kernel_inject=True,
                )
            else:
                # for bigscience/bloom, sharding is done while loading the model
                # so this is much slower and for this we need to create a
                # checkpoints json
                with TemporaryCheckpointsJSON(downloaded_model_path) as checkpoints_json:
                    self.model = deepspeed.init_inference(
                        self.model,
                        mp_size=world_size,
                        dtype=args.dtype,
                        checkpoint=checkpoints_json,
                        replace_with_kernel_inject=True,
                    )
        elif args.dtype == torch.bfloat16:
            # currently ds-inference only supports fp16 CUDA kernels :(
            raise NotImplementedError("bfloat16 is not yet supported")

        self.model = self.model.module
        self.input_device = torch.cuda.current_device()

        print_rank_n("Model loaded")
        dist.barrier()


# DeepSpeed MII class for making life easier for server deployment/CLI
class DSInferenceGRPCServer(Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # name for the deployment. this is used to get a reference to the GRPC
        # service, once deployed
        self.deployment_name = "ds_inference_grpc_server"

        downloaded_model_path = get_downloaded_model_path(args.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(downloaded_model_path)
        self.pad = self.tokenizer.pad_token_id

        if args.dtype in [torch.float16, torch.int8]:
            checkpoints_json = os.path.join(downloaded_model_path, "ds_inference_config.json")

            mii.deploy(
                task="text-generation",
                # should pass args.model_name but can't since the new
                # weights are not supported yet. So, this is a hack
                model="bigscience/bloom",
                deployment_name=self.deployment_name,
                model_path=downloaded_model_path,
                mii_config={
                    "dtype": get_str_dtype(args.dtype),
                    "tensor_parallel": 8,
                    "port_number": 50950,
                    "checkpoint_dict": json.load(open(checkpoints_json, "r")),
                },
            )
        elif args.dtype == torch.bfloat16:
            raise NotImplementedError("bfloat16 is not yet supported")

        # get the GRPC service launched in the above code
        self.model = mii.mii_query_handle(self.deployment_name)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        input_tokens = self.tokenizer(request.text).input_ids
        input_token_lengths = [len(x) for x in input_tokens]

        check_max_input_length(input_token_lengths, request.max_input_length)

        if request.stop_sequences is not None:
            raise NotImplementedError("DS-inference doesn't support stop_sequences")

        output_text = self.model.query({"query": request.text}, **get_filter_dict(request)).response

        output_text = [_ for _ in output_text]
        output_tokens = self.tokenizer(output_text).input_ids

        output_token_lengths = [len(x) for x in output_tokens]
        num_generated_tokens = [o - i for i, o in zip(input_token_lengths, output_token_lengths)]

        if request.remove_input_from_output:
            # the generate method's output includes input too. Remove input if
            # that is requested by the user
            output_tokens = [x[-i:] for x, i in zip(output_tokens, num_generated_tokens)]
            output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        return GenerateResponse(text=output_text, num_generated_tokens=num_generated_tokens)

    def shutdown(self) -> None:
        print_rank_n("shutting down")
        try:
            # try termination of the GRPC server. this is not guaranteed to be
            # successfull and the user might need to clear the GPU memory
            # manually by running mii.terminate(...) themselves.
            # MII is buggy and sometimes spits out an error in terminate
            mii.terminate(self.deployment_name)
        except Exception:
            pass
        exit()


class TemporaryCheckpointsJSON:
    def __init__(self, model_path: str):
        self.tmp_directory = "tmp"
        self.tmp_file = os.path.join(self.tmp_directory, "checkpoints.json")
        self.model_path = model_path

    def write_checkpoints_json(self, model_path: str) -> None:
        with io.open(self.tmp_file, "w", encoding="utf-8") as f:
            data = {"type": "BLOOM", "checkpoints": glob.glob(f"{model_path}/*.bin"), "version": 1.0}
            json.dump(data, f)

    def __enter__(self):
        run_rank_n(partial(os.makedirs, name=self.tmp_directory, exist_ok=True))
        run_rank_n(partial(self.write_checkpoints_json, model_path=self.model_path), barrier=True)
        return self.tmp_file

    def __exit__(self, type, value, traceback):
        return
