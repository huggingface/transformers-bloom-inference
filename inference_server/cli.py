import argparse
import json
import sys

from .model_handler import ModelDeployment
from .utils import get_argument_parser, parse_args, print_rank_0


def get_args() -> argparse.Namespace:
    parser = get_argument_parser()
    args = parse_args(parser)
    return args


def main() -> None:
    args = get_args()

    model = ModelDeployment(args, True)

    generate_kwargs = args.generate_kwargs

    while True:
        input_text = input("Input text: ")

        if input("change generate_kwargs? [y/n] ") == "y":
            while True:
                try:
                    generate_kwargs = json.loads(input("Generate kwargs: "))
                    break
                except Exception as e:
                    e_type, e_message, _ = sys.exc_info()
                    print("error =", e_type.__name__)
                    print("message =", e_message)
                    continue

        response = model.generate(text=[input_text], generate_kwargs=generate_kwargs)

        print_rank_0("Output text:", response.text[0])
        print_rank_0("Generated tokens:", response.num_generated_tokens[0])


if __name__ == "__main__":
    main()
