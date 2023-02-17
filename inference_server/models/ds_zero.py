from argparse import Namespace

import torch

import deepspeed
from transformers import AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig

from ..utils import get_world_size
from .model import Model, get_hf_model_class


class DSZeROModel(Model):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        config = AutoConfig.from_pretrained(args.model_name)

        train_micro_batch_size_per_gpu = 1
        train_batch_size = train_micro_batch_size_per_gpu * get_world_size()

        # try playing with these parameters, might improve throughput for you
        # hardware setup
        ds_config = {
            "fp16": {
                "enabled": args.dtype == torch.float16,
            },
            "bf16": {
                "enabled": args.dtype == torch.bfloat16,
            },
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": config.hidden_size * config.hidden_size,
                "stage3_prefetch_bucket_size": 0.9 * config.hidden_size * config.hidden_size,
                "stage3_param_persistence_threshold": 0,
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
            "wall_clock_breakdown": False,
        }

        if args.cpu_offload:
            ds_config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}

        # this tells from_pretrained to instantiate directly on gpus
        dschf = HfDeepSpeedConfig(ds_config)

        self.model = get_hf_model_class(args.model_class).from_pretrained(args.model_name, torch_dtype=args.dtype)
        self.model = self.model.eval()

        # convert model to a fully sharded model using ZeRO
        self.model = deepspeed.initialize(model=self.model, config_params=ds_config)[0]

        self.model.module.eval()
        self.model = self.model.module

        # this is the CUDA device for the current process. This will be used
        # later to identify the GPU on which to transfer tensors
        self.input_device = torch.cuda.current_device()

        self.post_init(args.model_name)
