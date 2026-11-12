"""Steady-state decode tok/s and prefill tok/s at several context lengths (8 warm-up steps excluded)."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from flash_next import FlashNext
m = FlashNext("/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw")
ids_all = [1000 + (i * 7919) % 60000 for i in range(32768)]
m.reset(); mx.eval(m.forward(ids_all[:600], last_only=True))  # compile warm-up
for n in [int(a) for a in (sys.argv[1:] or ["512", "4096", "12288", "24576"])]:
    m.reset()
    t0 = time.perf_counter()
    for s in range(0, n, 512):
        lg = m.forward(ids_all[s:s + 512], last_only=True); mx.eval(lg)
    tp = time.perf_counter() - t0
    tok = int(mx.argmax(lg[-1]).item())
    for _ in range(8):
        tok = int(mx.argmax(m.forward([tok])[-1]).item())
    t0 = time.perf_counter(); N = 48
    for _ in range(N):
        tok = int(mx.argmax(m.forward([tok])[-1]).item())
    td = time.perf_counter() - t0
    print(f"[active {mx.get_active_memory()/2**30:.1f} GiB, peak {mx.get_peak_memory()/2**30:.1f} GiB] ", end=""); print(f"context {n:6d}: prefill {n / tp:6.0f} tok/s ({tp:6.1f} s)   decode {N / td:5.1f} tok/s ({td / N * 1e3:5.1f} ms/token)", flush=True)
