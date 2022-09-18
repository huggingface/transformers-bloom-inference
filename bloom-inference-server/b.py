import torch
from ds_inference import DSInferenceGRPCServer

class Args:
    model_name = "microsoft/bloom-deepspeed-inference-fp16"
    dtype = torch.float16

model = DSInferenceGRPCServer(Args())
