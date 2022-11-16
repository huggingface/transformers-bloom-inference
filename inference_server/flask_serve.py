import argparse

from flask import Flask, request

from .constants import DS_INFERENCE, DS_ZERO
from .model_handler import ModelDeployment
from .utils import get_argument_parser, parse_args


def get_args() -> argparse.Namespace:
    parser = get_argument_parser()

    args = parse_args(parser)
    args.use_grpc_server = args.deployment_framework in [DS_INFERENCE, DS_ZERO]

    return args


app = Flask(__name__)
model = None


def do_prediction(query: str, **generate_kwargs):
    global model
    response = model.generate(text=query, generate_kwargs=generate_kwargs)
    return response.text[0]


@app.route('/predict')
def predict():
    all_args = request.args.to_dict()
    query = all_args.pop('query')
    return {'result': do_prediction(query, **all_args)}


def main() -> None:
    global model

    args = get_args()
    model = ModelDeployment(args, use_grpc_server=args.use_grpc_server,
                            cuda_visible_devices=args.cuda_visible_devices)

    app.run(host="0.0.0.0", port=args.serving_port)


if __name__ == "__main__":
    main()
