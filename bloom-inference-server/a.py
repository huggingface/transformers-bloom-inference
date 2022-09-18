import asyncio
import time
import torch
from ds_inference import DSInferenceGRPCServer
from utils import GenerateRequest

import nest_asyncio
nest_asyncio.apply()

class Args:
    model_name = "microsoft/bloom-deepspeed-inference-fp16"
    dtype = torch.float16

model = DSInferenceGRPCServer(Args())
time.sleep(10)
print("--------------------------------------------------------------------")

# --------------------------------------------------------------------
def f(request):
    return model.generate(request)

r = GenerateRequest(text=["hello"], max_new_tokens=20)
print(f(r))
# --------------------------------------------------------------------
async def g(request):
    return model.generate(request)

print(asyncio.run(g(r)))
# loop = asyncio.new_event_loop()
# print(loop.run_until_complete(g(r)))
# loop.close()
# --------------------------------------------------------------------
