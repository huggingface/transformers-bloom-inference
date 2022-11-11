import glob
import io
import json
import os
from argparse import Namespace
from functools import partial

import torch
import torch.distributed as dist

import deepspeed
from transformers import AutoConfig, AutoTokenizer

from ..utils import print_rank_n, run_rank_n
from .model import Model, get_downloaded_model_path, get_hf_model_class, load_tokenizer


# basic DeepSpeed inference model class for benchmarking
class DSInferenceModel(Model):
    def __init__(self, args: Namespace) -> None:
        print_rank_n("Loading model...")

        super().__init__(args)

        world_size = int(os.getenv("WORLD_SIZE", "1"))

        downloaded_model_path = get_downloaded_model_path(args.model_name)

        self.tokenizer = load_tokenizer(downloaded_model_path)
        self.pad = self.tokenizer.pad_token_id

        # create dummy tensors for allocating space which will be filled with
        # the actual weights while calling deepspeed.init_inference in the
        # following code
        with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
            self.model = get_hf_model_class(args.model_class).from_config(
                AutoConfig.from_pretrained(downloaded_model_path), torch_dtype=torch.bfloat16
            )
        self.model = self.model.eval()

        if args.dtype in [torch.float16, torch.int8]:
            # We currently support the weights provided by microsoft (which are
            # pre-sharded)
            if args.use_pre_sharded_checkpoints:
                checkpoints_json = os.path.join(downloaded_model_path, "ds_inference_config.json")

                self.model = deepspeed.init_inference(
                    self.model,
                    mp_size=world_size,
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
                        mp_size=world_size,
                        dtype=args.dtype,
                        checkpoint=checkpoints_json,
                        replace_with_kernel_inject=True,
                    )
        elif args.dtype == torch.bfloat16:
            # currently ds-inference only supports fp16 CUDA kernels :(
            raise NotImplementedError("bfloat16 is not yet supported")

        self.model = self.model.module
        self.input_device = torch.cuda.current_device()

        print_rank_n("Model loaded")
        dist.barrier()


class TemporaryCheckpointsJSON:
    def __init__(self, model_path: str):
        self.tmp_directory = "tmp"
        self.tmp_file = os.path.join(self.tmp_directory, "checkpoints.json")
        self.model_path = model_path

    def write_checkpoints_json(self, model_path: str) -> None:
        with io.open(self.tmp_file, "w", encoding="utf-8") as f:
            data = {"type": "BLOOM", "checkpoints": glob.glob(f"{model_path}/*.bin"), "version": 1.0}
            json.dump(data, f)

    def __enter__(self):
        run_rank_n(partial(os.makedirs, name=self.tmp_directory, exist_ok=True))
        run_rank_n(partial(self.write_checkpoints_json, model_path=self.model_path), barrier=True)
        return self.tmp_file

    def __exit__(self, type, value, traceback):
        return
