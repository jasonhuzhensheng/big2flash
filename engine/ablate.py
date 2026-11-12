"""Decode-step time with one component replaced by zeros at a time (outputs become wrong; timing only)."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
import flash_next as fn
m = fn.FlashNext("/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw")
H = m.cfg.hidden
def step_time(n=25):
    m.reset(); mx.eval(m.forward(list(range(1000, 1064))))
    tok = 1234
    for _ in range(3): tok = int(mx.argmax(m.forward([tok])[-1]).item())
    t = time.perf_counter()
    for _ in range(n): tok = int(mx.argmax(m.forward([tok])[-1]).item())
    return (time.perf_counter() - t) / n * 1e3
base = step_time(); print(f"full step                      {base:6.1f} ms", flush=True)
zero = lambda *a, **k: mx.zeros((1, H))
for name, attr, repl in [
    ("routed experts", "experts", lambda x, idx, sc, p: mx.zeros((1, H)) + 0 * sc.sum()),
    ("whole MoE block", "moe", zero),
    ("linear attention (36)", "linear_attn", zero),
    ("full attention (12)", "attention", zero),
    ("PLE", "ple", lambda h, ids, i: mx.zeros_like(h)),
]:
    orig = getattr(fn.FlashNext, attr)
    setattr(fn.FlashNext, attr, lambda self, *a, _r=repl: _r(*a))
    t = step_time(); print(f"without {name:22s} {t:6.1f} ms  (component ~{base - t:5.1f} ms)", flush=True)
    setattr(fn.FlashNext, attr, orig)
orig = fn.FlashNext.gated_residual
fn.FlashNext.gated_residual = lambda self, h, name, combine=True: (h[:, :H], mx.ones((1, 4))) if combine else h[:, :H]
t = step_time(); print(f"without hyper-connections      {t:6.1f} ms  (component ~{base - t:5.1f} ms)")
fn.FlashNext.gated_residual = orig
