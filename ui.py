import argparse

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.routing import APIRoute, Mount
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoConfig, AutoTokenizer
from uvicorn import run


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="model")
    group.add_argument("--model_name", type=str, required=True, help="model name")

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--ui_host", type=str, default="127.0.0.1", help="host address for UI")
    group.add_argument("--ui_port", type=int, default=5001, help="port number for UI")
    group.add_argument("--server_host", type=str, default="127.0.0.1", help="host address for generation server")
    group.add_argument("--server_port", type=int, default=5000, help="port number for generation server")

    return parser.parse_args()


class Server:
    def __init__(self, args: argparse.Namespace):
        self.templates = Jinja2Templates(directory="templates")
        self.ui_host = args.ui_host
        self.ui_port = args.ui_port
        self.server_host = args.server_host
        self.server_port = args.server_port
        self.workers = 1
        self.is_encoder_decoder = AutoConfig.from_pretrained(args.model_name).is_encoder_decoder

        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")

        self.app = FastAPI(
            routes=[
                APIRoute("/", self.homepage, methods=["GET"], response_class=HTMLResponse),
                APIRoute("/generate/", self.generate, methods=["POST"]),
                Mount("/static/", StaticFiles(directory="static"), name="static"),
            ],
            timeout=600,
        )

        self.prefix_checkpoints_list = None

    def homepage(self, request: Request) -> HTMLResponse:
        if self.is_encoder_decoder:
            return self.templates.TemplateResponse("encoder_decoder.html", {"request": request})
        else:
            return self.templates.TemplateResponse("decoder.html", {"request": request})

    def generate(self, request: dict) -> JSONResponse:
        response = requests.post(f"http://{self.server_host}:{self.server_port}/generate", json=request, verify=False)
        return JSONResponse(content=response.json())

    def run(self):
        # get around CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        run(self.app, host=self.ui_host, port=self.ui_port, workers=self.workers)


def main() -> None:
    Server(get_args()).run()


if __name__ == "__main__":
    main()
