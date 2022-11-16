"""
Copyright 2022 The Microsoft DeepSpeed Team
"""
import argparse
import asyncio
import os
import subprocess
import time
from typing import List

import grpc
from mii.server_client import MIIServerClient
from transformers import AutoTokenizer

from ..constants import DS_INFERENCE, DS_ZERO
from ..models import get_downloaded_model_path, get_model_class, load_tokenizer
from ..utils import (
    GenerateResponse,
    TokenizeRequest,
    TokenizeResponse,
    create_generate_request,
    get_str_dtype,
    print_rank_n,
)
from .grpc_utils.pb import generation_pb2, generation_pb2_grpc


class ModelDeployment(MIIServerClient):
    def __init__(self, args: argparse.Namespace, use_grpc_server: bool = False, cuda_visible_devices: List[int] = [0]):
        self.cuda_visible_devices = cuda_visible_devices
        self.num_gpus = len(self.cuda_visible_devices)
        self.use_grpc_server = use_grpc_server

        if self.use_grpc_server:
            self.tokenizer = load_tokenizer(get_downloaded_model_path(args.model_name))

            self.initialize_ports()

            self.dtype_proto_field = {
                str: "svalue",
                int: "ivalue",
                float: "fvalue",
                bool: "bvalue",
            }

            self._initialize_service(args)
            self._wait_until_server_is_live()

            self.asyncio_loop = asyncio.get_event_loop()
            self._initialize_grpc_client()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.cuda_visible_devices))
            self.model = get_model_class(args.deployment_framework)(args)

    def initialize_ports(self):
        self.ports = []
        for i in range(self.num_gpus):
            self.ports.append(50950 + self.cuda_visible_devices[i])

    def _wait_until_server_is_live(self):
        sockets_open = False
        while not sockets_open:
            sockets_open = self._is_socket_open(self.ports[0])
            process_alive = self._is_server_process_alive()
            if not process_alive:
                raise RuntimeError("server crashed for some reason, unable to proceed")
            time.sleep(4)
            print_rank_n("waiting for server to start...")
        print_rank_n(f"server has started on {self.ports[0]}")

    def dict_to_proto(self, generate_kwargs: dict) -> dict:
        result = {}
        for k, v in generate_kwargs.items():
            if v is not None:
                x = generation_pb2.Value()
                setattr(x, self.dtype_proto_field[type(v)], v)
                result[k] = x

        return result

    def _initialize_service(self, args: argparse.Namespace):
        if self._is_socket_open(self.ports[0]):
            raise RuntimeError(
                f"Server is already running on port {self.ports}, please shutdown or use different port."
            )

        if args.deployment_framework in [DS_INFERENCE, DS_ZERO]:
            ports = " ".join(map(str, self.ports))

            cmd = f"inference_server.model_handler.launch --model_name {args.model_name} --deployment_framework {args.deployment_framework} --dtype {get_str_dtype(args.dtype)} --port {ports} --model_class {args.model_class}"

            if args.max_batch_size is not None:
                cmd += f" --max_batch_size {args.max_batch_size}"
            if args.max_input_length is not None:
                cmd += f" --max_input_length {args.max_input_length}"

            master_port = 29500 + min(self.cuda_visible_devices)

            cuda_visible_devices = ",".join(map(str, self.cuda_visible_devices))

            cmd = f"deepspeed --master_port {master_port} --include localhost:{cuda_visible_devices} --module {cmd}"
        else:
            raise NotImplementedError(f"unsupported deployment_framework: {args.deployment_framework}")

        cmd = cmd.split(" ")
        self.process = subprocess.Popen(cmd)

    def _initialize_grpc_client(self):
        self.stubs = []
        for i in self.ports:
            channel = grpc.aio.insecure_channel(f"localhost:{i}")
            stub = generation_pb2_grpc.GenerationServiceStub(channel)
            self.stubs.append(stub)

    # runs task in parallel and return the result from the first task
    async def _query_in_tensor_parallel(self, text: List[str], generate_kwargs: dict):
        responses = []
        for i in range(self.num_gpus):
            responses.append(self.asyncio_loop.create_task(self._request_async_response(i, text, generate_kwargs)))

        await responses[0]
        return responses[0]

    async def _request_async_response(self, stub_id: int, text: List[str], generate_kwargs: dict):
        req = generation_pb2.GenerationRequest(texts=text, generate_kwargs=generate_kwargs)
        response = await self.stubs[stub_id].Generate(req)
        return response

    def generate(self, **kwargs) -> GenerateResponse:
        if self.use_grpc_server:
            if "request" in kwargs:
                text = kwargs["request"].text
                generate_kwargs = kwargs["request"].get_generate_kwargs()
            else:
                text = kwargs["text"]
                generate_kwargs = kwargs["generate_kwargs"]

            generate_kwargs = self.dict_to_proto(generate_kwargs)

            response = self.asyncio_loop.run_until_complete(
                self._query_in_tensor_parallel(text, generate_kwargs)
            ).result()

            if response.error:
                raise Exception(response.error)
            else:
                return GenerateResponse(
                    text=[r for r in response.texts], num_generated_tokens=[n for n in response.num_generated_tokens]
                )
        else:
            if "request" in kwargs:
                request = kwargs["request"]
            else:
                request = create_generate_request(**kwargs)

            response = self.model.generate(request)

            if isinstance(response, Exception):
                raise response
            else:
                return response

    def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        if self.use_grpc_server:
            response = self.tokenizer(request.text, padding=request.padding)
            response = TokenizeResponse(token_ids=response.input_ids, attention_mask=response.attention_mask)
        else:
            response = self.model.tokenize(request)

        return response

    def _request_response(self):
        raise NotImplementedError("This method should not be implemented")

    def query(self):
        raise NotImplementedError("This method should not be implemented")
