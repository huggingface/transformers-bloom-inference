from argparse import Namespace

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import print_rank_n
from .model import Model, get_downloaded_model_path, get_hf_model_class, load_tokenizer


class HFAccelerateModel(Model):
    def __init__(self, args: Namespace) -> None:
        print_rank_n("Loading model...")

        super().__init__(args)

        downloaded_model_path = get_downloaded_model_path(args.model_name)

        self.tokenizer = load_tokenizer(downloaded_model_path)
        self.pad = self.tokenizer.pad_token_id

        kwargs = {"pretrained_model_name_or_path": downloaded_model_path, "device_map": "auto"}

        if len(args.cuda_visible_devices) > 1:
            kwargs["device_map"] = "balanced_low_0"

        if args.dtype == torch.int8:
            # using LLM.int8()
            kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = args.dtype

        # this is the CUDA device for the current process. This will be used
        # later to identify the GPU on which to transfer tensors
        self.model = get_hf_model_class(args.model_class).from_pretrained(**kwargs)

        self.model.requires_grad_(False)
        self.model.eval()
        self.input_device = "cuda:0"

        print_rank_n("Model loaded")
