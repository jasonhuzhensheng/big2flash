"""How close is decode to the memory-bandwidth limit? Bare pipelined mlx-lm loop, no server.

usage: decode_limit.py MODEL_DIR [--fp16]
Prints weight bytes actually resident, measured copy bandwidth, the implied ceiling (bytes / bandwidth),
and measured decode speed at a short context. --fp16 casts every non-quantized tensor to float16
(M1/M2 have no bf16 matrix units).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from quiet import wait_quiet
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def bandwidth():
    a = mx.zeros((1 << 28,), dtype=mx.float32)  # 1 GiB
    mx.eval(a)
    best = 0
    for _ in range(6):
        t = time.perf_counter()
        b = a + 1
        mx.eval(b)
        best = max(best, 2 * a.nbytes / (time.perf_counter() - t))
        del b
    return best


path = sys.argv[1]
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])  # before loading: unwired weights read at ~40 GB/s
model, tok = load(path)
if "--fp16" in sys.argv:
    model.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, model.parameters()))
mx.eval(model.parameters())
weights = mx.get_active_memory()
wait_quiet()
bw = bandwidth()
print(f"weights resident {weights / 1e9:.2f} GB, bandwidth {bw / 1e9:.0f} GB/s, "
      f"ceiling {bw / weights:.1f} tok/s")

cache = make_prompt_cache(model)
ids = mx.array(tok.encode("Write a long story about a lighthouse keeper.") * 8)[None]
logits = model(ids, cache=cache)
y = mx.argmax(logits[0, -1])
mx.eval(y)
for n in (32, 256):
    wait_quiet()
    t = time.perf_counter()
    nxt = None
    for i in range(n):
        nxt = mx.argmax(model(y.reshape(1, 1), cache=cache)[0, -1])
        mx.async_eval(nxt)
        y.item()
        y = nxt
    y.item()
    dt = time.perf_counter() - t
    print(f"decode {n} tokens: {n / dt:.1f} tok/s ({n / dt / (bw / weights) * 100:.0f}% of ceiling)")
