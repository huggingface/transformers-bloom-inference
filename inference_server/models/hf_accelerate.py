from argparse import Namespace

import torch

from ..utils import get_world_size
from .model import Model, get_hf_model_class


class HFAccelerateModel(Model):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        kwargs = {"pretrained_model_name_or_path": args.model_name, "device_map": "auto"}

        if get_world_size() > 1:
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

        self.post_init(args.model_name)
