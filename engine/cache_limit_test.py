"""Decode speed at short context under different MLX buffer-cache limits."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from flash_next import FlashNext
m = FlashNext("/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw")
for lim in (2, 6, 10, None):
    mx.set_cache_limit((lim << 30) if lim else mx.device_info()["memory_size"])
    m.reset(); mx.eval(m.forward(list(range(1000, 1512)), last_only=True))
    tok = 1234
    for _ in range(8): tok = int(mx.argmax(m.forward([tok])[-1]).item())
    t0 = time.perf_counter()
    for _ in range(32): tok = int(mx.argmax(m.forward([tok])[-1]).item())
    dt = (time.perf_counter() - t0) / 32
    print(f"cache limit {str(lim) + ' GiB' if lim else 'none':8s}: decode {1/dt:5.1f} tok/s  cache {mx.get_cache_memory()/2**30:5.2f} GiB  active {mx.get_active_memory()/2**30:.1f} GiB", flush=True)
