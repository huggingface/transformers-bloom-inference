import argparse

from .models import get_downloaded_model_path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model to use",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()
    get_downloaded_model_path(args.model_name)


if __name__ == "__main__":
    main()
