import asyncio
import torch
from ds_inference import DSInferenceGRPCServer
from utils import GenerateRequest

class Args:
    model_name = "microsoft/bloom-deepspeed-inference-fp16"
    dtype = torch.float16

model = DSInferenceGRPCServer(Args())

async def f(requuest):
    model.generate(requuest)

loop = asyncio.new_event_loop()
r = GenerateRequest(text=["hello"], max_new_tokens=20)
print(loop.run_until_complete(f(r)))
loop.close()
