import json
import os
from functools import partial

import torch
from flask import Flask, request
from pydantic import BaseModel
from requests.exceptions import HTTPError

from ds_inference import DSInferenceGRPCServer
from utils import (HF_ACCELERATE, GenerateRequest, TokenizeRequest,
                   get_exception_response, get_num_tokens_to_generate,
                   get_torch_dtype, parse_generate_kwargs, run_and_log_time)


class QueryID(BaseModel):
    generate_query_id: int = 0
    tokenize_query_id: int = 0


# placeholder class for getting args. gunicorn does not allow passing args to a
# python script via ArgumentParser
class Args:
    deployment_framework: str = os.getenv(
        "DEPLOYMENT_FRAMEWORK", HF_ACCELERATE)
    model_name: str = os.getenv("MODEL_NAME")
    dtype: torch.dtype = get_torch_dtype(os.getenv("DTYPE"))
    generate_kwargs: dict = parse_generate_kwargs(
        json.loads(
            os.getenv(
                "GENERATE_KWARGS",
                '{"min_length": 100, "max_new_tokens": 100, "do_sample": false}'
            )
        )
    )


# ------------------------------------------------------
args = Args()
app = Flask(__name__)
model = DSInferenceGRPCServer(args)
query_ids = QueryID()
# ------------------------------------------------------


@app.route("/query_id/", methods=["GET"])
def query_id():
    return query_ids


@app.route("/tokenize/", methods=["POST"])
def tokenize():
    try:
        x = request.get_json()
        x = TokenizeRequest(**x)

        response, total_time_taken = run_and_log_time(
            partial(model.tokenize, request=x)
        )

        response.query_id = query_ids.tokenize_query_id
        query_ids.tokenize_query_id += 1
        response.total_time_taken = "{:.2f} msecs".format(
            total_time_taken * 1000)

        return response
    except Exception:
        response = get_exception_response(
            query_ids.tokenize_query_id, x.method)
        query_ids.tokenize_query_id += 1
        raise HTTPError(response)


@app.route("/generate/", methods=["POST"])
def generate():
    try:
        x = request.get_json()
        x = GenerateRequest(**x)
        x.preprocess()

        x.max_new_tokens = get_num_tokens_to_generate(
            x.max_new_tokens, args.allowed_max_new_tokens)

        response, total_time_taken = run_and_log_time(
            partial(model.generate, request=x)
        )

        response.query_id = query_ids.generate_query_id
        query_ids.generate_query_id += 1
        response.total_time_taken = "{:.2f} msecs".format(
            total_time_taken * 1000)

        return response
    except Exception:
        response = get_exception_response(
            query_ids.generate_query_id, x.method)
        query_ids.generate_query_id += 1
        raise HTTPError(response)
