import argparse
import json
import os

import torch
from transformers import AutoTokenizer

import mii
from utils import (
    GenerateRequest,
    GenerateResponse,
    Model,
    get_downloaded_model_path,
    get_filter_dict,
    get_str_dtype,
    print_rank_n,
    mii_query_handle
)


class DSInferenceGRPCServer(Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # name for the deployment. this is used to get a reference to the GRPC
        # service, once deployed
        self.deployment_name = "ds_inference_grpc_server"

        downloaded_model_path = get_downloaded_model_path(args.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(downloaded_model_path)
        self.pad = self.tokenizer.pad_token_id

        if (args.dtype in [torch.float16, torch.int8]):
            checkpoints_json = os.path.join(
                downloaded_model_path, "ds_inference_config.json")

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
                    "checkpoint_dict": json.load(open(checkpoints_json, "r"))
                }
            )
        elif (args.dtype == torch.bfloat16):
            raise NotImplementedError("bfloat16 is not yet supported")

        # get the GRPC service launched in the above code
        self.model = mii_query_handle(self.deployment_name)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        output_text = self.model.query(
            {"query": request.text},
            **get_filter_dict(request)
        ).response

        output_text = [_ for _ in output_text]

        # Remove input from output
        input_tokens = self.tokenizer(request.text).input_ids
        output_tokens = self.tokenizer(output_text).input_ids

        input_token_lengths = [len(x) for x in input_tokens]
        output_token_lengths = [len(x) for x in output_tokens]
        num_generated_tokens = [
            o - i for i, o in zip(input_token_lengths, output_token_lengths)]

        if (request.remove_input_from_output):
            # the generate method's output includes input too. Remove input if
            # that is requested by the user
            output_tokens = [x[-i:]
                             for x, i in zip(output_tokens, num_generated_tokens)]
            output_text = self.tokenizer.batch_decode(
                output_tokens, skip_special_tokens=True)

        return GenerateResponse(
            text=output_text,
            num_generated_tokens=num_generated_tokens
        )

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
