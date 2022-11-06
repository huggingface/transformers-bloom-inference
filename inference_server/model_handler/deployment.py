import argparse
import asyncio
import subprocess
from typing import List

import grpc
from mii.server_client import MIIServerClient
from transformers import AutoTokenizer

from ..constants import DS_INFERENCE, DS_ZERO
from ..models import get_downloaded_model_path, get_model_class
from ..utils import GenerateResponse, TokenizeRequest, TokenizeResponse, create_generate_request, get_str_dtype
from .grpc_utils.pb import generation_pb2, generation_pb2_grpc


class ModelDeployment(MIIServerClient):
    def __init__(self, args: argparse.Namespace, use_grpc_server: bool = False, port: int = 50950, num_gpus: int = 1):
        self.num_gpus = num_gpus
        self.use_grpc_server = use_grpc_server

        if self.use_grpc_server:
            self.tokenizer = AutoTokenizer.from_pretrained(get_downloaded_model_path(args.model_name))

            self.port_number = port

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
            self.model = get_model_class(args.deployment_framework)(args)

    def dict_to_proto(self, generate_kwargs: dict) -> dict:
        result = {}
        for k, v in generate_kwargs.items():
            if v is not None:
                x = generation_pb2.Value()
                setattr(x, self.dtype_proto_field[type(v)], v)
                result[k] = x

        return result

    def _initialize_service(self, args: argparse.Namespace):
        if self._is_socket_open(self.port_number):
            raise RuntimeError(
                f"Server is already running on port {self.port_number}, please shutdown or use different port."
            )

        cmd = f"inference_server.model_handler.launch --model_name {args.model_name} --deployment_framework {args.deployment_framework} --dtype {get_str_dtype(args.dtype)} --port {self.port_number}"

        if args.deployment_framework in [DS_INFERENCE, DS_ZERO]:
            cmd = f"deepspeed --num_gpus {self.num_gpus} --module {cmd}"
        else:
            raise NotImplementedError(f"unsupported deployment_framework: {args.deployment_framework}")

        cmd = cmd.split(" ")
        self.process = subprocess.Popen(cmd)

    def _initialize_grpc_client(self):
        channels = []
        self.stubs = []
        for i in range(self.num_gpus):
            channel = grpc.aio.insecure_channel(f"localhost:{self.port_number + i}")
            stub = generation_pb2_grpc.GenerationServiceStub(channel)
            channels.append(channel)
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
                request = kwargs["request"]
                text = request.text
                generate_kwargs = request.get_generate_kwargs()
                generate_kwargs = self.dict_to_proto(generate_kwargs)
            else:
                text = kwargs["text"]
                generate_kwargs = self.dict_to_proto(kwargs["generate_kwargs"])

            response = self.asyncio_loop.run_until_complete(
                self._query_in_tensor_parallel(text, generate_kwargs)
            ).result()

            response = GenerateResponse(
                text=[r for r in response.texts], num_generated_tokens=[n for n in response.num_generated_tokens]
            )
        else:
            if "request" in kwargs:
                request = kwargs["request"]
            else:
                request = create_generate_request(**kwargs)

            response = self.model.generate(request)

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
