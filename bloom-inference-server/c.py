import asyncio
import time
from utils import GenerateRequest

import nest_asyncio
nest_asyncio.apply()

from .b import model

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

for i in range(100):
    s = time.time()
    print(asyncio.run(g(r)))
    print(time.time() - s, "c")
# loop = asyncio.new_event_loop()
# print(loop.run_until_complete(g(r)))
# loop.close()
# --------------------------------------------------------------------
