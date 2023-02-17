import os
from functools import partial

from flask import Flask, request
from flask_api import status
from pydantic import BaseModel

from .constants import HF_ACCELERATE
from .model_handler.deployment import ModelDeployment
from .utils import (
    ForwardRequest,
    GenerateRequest,
    TokenizeRequest,
    get_exception_response,
    get_num_tokens_to_generate,
    get_torch_dtype,
    parse_bool,
    run_and_log_time,
)


class QueryID(BaseModel):
    generate_query_id: int = 0
    tokenize_query_id: int = 0
    forward_query_id: int = 0


# placeholder class for getting args. gunicorn does not allow passing args to a
# python script via ArgumentParser
class Args:
    def __init__(self) -> None:
        self.deployment_framework = os.getenv("DEPLOYMENT_FRAMEWORK", HF_ACCELERATE)
        self.model_name = os.getenv("MODEL_NAME")
        self.model_class = os.getenv("MODEL_CLASS")
        self.dtype = get_torch_dtype(os.getenv("DTYPE"))
        self.allowed_max_new_tokens = int(os.getenv("ALLOWED_MAX_NEW_TOKENS", 100))
        self.max_input_length = int(os.getenv("MAX_INPUT_LENGTH", 512))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 4))
        self.debug = parse_bool(os.getenv("DEBUG", "false"))


# ------------------------------------------------------
args = Args()
model = ModelDeployment(args, True)
query_ids = QueryID()
app = Flask(__name__)
# ------------------------------------------------------


@app.route("/query_id/", methods=["GET"])
def query_id():
    return query_ids.dict(), status.HTTP_200_OK


@app.route("/tokenize/", methods=["POST"])
def tokenize():
    try:
        x = request.get_json()
        x = TokenizeRequest(**x)

        response, total_time_taken = run_and_log_time(partial(model.tokenize, request=x))

        response.query_id = query_ids.tokenize_query_id
        query_ids.tokenize_query_id += 1
        response.total_time_taken = "{:.2f} msecs".format(total_time_taken * 1000)

        return response.dict(), status.HTTP_200_OK
    except Exception:
        response = get_exception_response(query_ids.tokenize_query_id, args.debug)
        query_ids.tokenize_query_id += 1
        return response, status.HTTP_500_INTERNAL_SERVER_ERROR


@app.route("/generate/", methods=["POST"])
def generate():
    try:
        x = request.get_json()
        x = GenerateRequest(**x)

        x.max_new_tokens = get_num_tokens_to_generate(x.max_new_tokens, args.allowed_max_new_tokens)

        response, total_time_taken = run_and_log_time(partial(model.generate, request=x))

        response.query_id = query_ids.generate_query_id
        query_ids.generate_query_id += 1
        response.total_time_taken = "{:.2f} secs".format(total_time_taken)

        return response.dict(), status.HTTP_200_OK
    except Exception:
        response = get_exception_response(query_ids.generate_query_id, args.debug)
        query_ids.generate_query_id += 1
        return response, status.HTTP_500_INTERNAL_SERVER_ERROR


@app.route("/forward/", methods=["POST"])
def forward():
    try:
        x = request.get_json()
        x = ForwardRequest(**x)

        if len(x.conditioning_text) != len(x.response):
            raise Exception("unequal number of elements in conditioning_text and response arguments")

        response, total_time_taken = run_and_log_time(partial(model.forward, request=x))

        response.query_id = query_ids.forward_query_id
        query_ids.forward_query_id += 1
        response.total_time_taken = "{:.2f} secs".format(total_time_taken)

        return response.dict(), status.HTTP_200_OK
    except Exception:
        response = get_exception_response(query_ids.forward_query_id, args.debug)
        query_ids.forward_query_id += 1
        return response, status.HTTP_500_INTERNAL_SERVER_ERROR
