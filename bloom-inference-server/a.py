import asyncio
import torch
from ds_inference import DSInferenceGRPCServer
from utils import GenerateRequest

class Args:
    model_name = "microsoft/bloom-deepspeed-inference-fp16"
    dtype = torch.float16

model = DSInferenceGRPCServer(Args())

print(asyncio.run(model.generate(GenerateRequest(text=["hello"], max_new_tokens=20))))