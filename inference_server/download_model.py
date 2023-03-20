import argparse

from transformers import AutoConfig, AutoTokenizer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model to use",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        required=True,
        help="model class to use",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()
    print("downloading", args.model_name)
    AutoConfig.from_pretrained(args.model_name)
    AutoTokenizer.from_pretrained(args.model_name)
    args.model_class.from_pretrained(args.model_name)


if __name__ == "__main__":
    main()
