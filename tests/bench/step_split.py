"""One decode step: CPU graph-building time vs total, stock vs qmm4."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from quiet import wait_quiet
import qmm4_experiment as qmm4
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
m, _ = load(os.path.expanduser("~/Library/Application Support/big2flash/models/Qwen3.8-27B-4bit-MTP-fp16"))
cache = make_prompt_cache(m)
mx.eval(m(mx.array([list(range(100, 600))]), cache=cache))
def trim(n):
    for c in cache:
        if hasattr(c, "trim"): c.trim(n)
for label, patch in [("stock", False), ("qmm4", True), ("stock", False), ("qmm4", True)]:
    nn.QuantizedLinear.__call__ = qmm4._ql_call if patch else qmm4._ql_orig
    for T in (1, 2):
        x = mx.array([[1000 + i for i in range(T)]])
        for _ in range(3): mx.eval(m(x, cache=cache)); trim(T)
        wait_quiet()
        build = tot = 0
        for _ in range(20):
            t0 = time.perf_counter(); y = m(x, cache=cache); t1 = time.perf_counter(); mx.eval(y); t2 = time.perf_counter()
            trim(T); build += t1 - t0; tot += t2 - t0
        print(f"{label} T={T}: build {build/20*1000:.1f} ms, total {tot/20*1000:.1f} ms", flush=True)
