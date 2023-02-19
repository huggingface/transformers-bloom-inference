import glob
import io
import json
import os
from argparse import Namespace
from functools import partial

import torch

import deepspeed
from huggingface_hub import try_to_load_from_cache
from transformers import AutoConfig

from ..utils import get_world_size, run_rank_n
from .model import Model, get_hf_model_class


# basic DeepSpeed inference model class for benchmarking
class DSInferenceModel(Model):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        # create dummy tensors for allocating space which will be filled with
        # the actual weights while calling deepspeed.init_inference in the
        # following code
        with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
            self.model = get_hf_model_class(args.model_class).from_config(
                AutoConfig.from_pretrained(args.model_name), torch_dtype=torch.bfloat16
            )
        self.model = self.model.eval()

        downloaded_model_path = get_model_path(args.model_name)

        if args.dtype in [torch.float16, torch.int8]:
            # We currently support the weights provided by microsoft (which are
            # pre-sharded)
            checkpoints_json = os.path.join(downloaded_model_path, "ds_inference_config.json")

            if os.path.isfile(checkpoints_json):
                self.model = deepspeed.init_inference(
                    self.model,
                    mp_size=get_world_size(),
                    base_dir=downloaded_model_path,
                    dtype=args.dtype,
                    checkpoint=checkpoints_json,
                    replace_with_kernel_inject=True,
                )
            else:
                # for bigscience/bloom, sharding is done while loading the model
                # so this is much slower and for this we need to create a
                # checkpoints json
                with TemporaryCheckpointsJSON(downloaded_model_path) as checkpoints_json:
                    self.model = deepspeed.init_inference(
                        self.model,
                        mp_size=get_world_size(),
                        base_dir=downloaded_model_path,
                        dtype=args.dtype,
                        checkpoint=checkpoints_json,
                        replace_with_kernel_inject=True,
                    )
        elif args.dtype == torch.bfloat16:
            # currently ds-inference only supports fp16 CUDA kernels :(
            raise NotImplementedError("bfloat16 is not yet supported")

        self.model = self.model.module
        self.input_device = torch.cuda.current_device()

        self.post_init(args.model_name)


class TemporaryCheckpointsJSON:
    def __init__(self, model_path: str):
        self.tmp_directory = "tmp"
        self.tmp_file = os.path.join(self.tmp_directory, "checkpoints.json")
        self.model_path = model_path

    def write_checkpoints_json(self) -> None:
        print(self.model_path)
        with io.open(self.tmp_file, "w", encoding="utf-8") as f:
            data = {"type": "BLOOM", "checkpoints": glob.glob(f"{self.model_path}/*.bin"), "version": 1.0}
            json.dump(data, f)

    def __enter__(self):
        run_rank_n(os.makedirs, barrier=True)(self.tmp_directory, exist_ok=True)
        run_rank_n(self.write_checkpoints_json, barrier=True)()
        return self.tmp_file

    def __exit__(self, type, value, traceback):
        return


def get_model_path(model_name: str):
    try:
        config_file = "config.json"

        # will fall back to HUGGINGFACE_HUB_CACHE
        config_path = try_to_load_from_cache(model_name, config_file, cache_dir=os.getenv("TRANSFORMERS_CACHE"))

        if config_path is None:
            # treat the model name as an explicit model path
            return model_name
        else:
            return os.path.dirname(config_path)
    except:
        # treat the model name as an explicit model path
        return model_name
