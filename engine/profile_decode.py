"""Where a decode step's time goes: graph building, GPU evaluation, n-gram reads."""
import os, sys, time
import mlx.core as mx
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from flash_next import FlashNext
M = "/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw"
m = FlashNext(M)
ids = list(range(1000, 1064))
mx.eval(m.forward(ids))
tok = 1234
for _ in range(5):
    tok = int(mx.argmax(m.forward([tok])[-1]).item())
N = 20
tb = te = 0.0
for _ in range(N):
    t0 = time.perf_counter(); lg = m.forward([tok]); t1 = time.perf_counter()
    tok = int(mx.argmax(lg[-1]).item()); t2 = time.perf_counter()
    tb += t1 - t0; te += t2 - t1
print(f"decode step: build {tb/N*1e3:.1f} ms + eval {te/N*1e3:.1f} ms = {(tb+te)/N*1e3:.1f} ms ({N/(tb+te):.1f} tok/s)")
rng = np.random.default_rng(0)
for n in (16, 16, 160):
    r = rng.integers(0, m.ngram.rows_n, n)
    t0 = time.perf_counter(); m.ngram.rows(r); print(f"ngram rows x{n}: {(time.perf_counter()-t0)*1e3:.1f} ms")
# GPU-only: rebuild the same step, eval twice to separate compile/first-run effects
st = [{k: v for k, v in s.items()} for s in m.state]
for _ in range(3):
    lg = m.forward([tok]); mx.eval(lg)
t0 = time.perf_counter()
for _ in range(10):
    lg = m.forward([tok]); mx.eval(lg)
print(f"forward+eval (no argmax sync): {(time.perf_counter()-t0)/10*1e3:.1f} ms")
