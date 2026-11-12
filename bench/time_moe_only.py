"""Kernel-only timing: routes precomputed, 48 dependent layers per eval, 8 random real-shaped banks."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
import moe_kernel as mk

K, E, H, I = 10, 512, mk.H, mk.I
def bank():
    def p(o, i):
        g = (E, o, i // 64)
        t = (mx.random.randint(0, 1 << 30, (E, o, i // 8)).astype(mx.uint32),
             (mx.random.uniform(shape=g) * 1e-3).astype(mx.bfloat16), mx.zeros(g, mx.bfloat16))
        mx.eval(*t); return t
    return (p(I, H), p(I, H), p(H, I))
banks = [bank() for _ in range(8)]
per_token = 48 * K * sum(a.nbytes for q in banks[0] for a in q) / E
routes = []
for l in range(48):
    idx = mx.random.permutation(E)[:K].astype(mx.uint32); g = mx.softmax(mx.random.normal((K,)))
    mx.eval(idx, g); routes.append((idx, g))

def run(fn):
    x = mx.random.normal((H,))
    for l in range(48):
        x = fn(x, *routes[l], *banks[l % 8]) * 1e-3 + x * 0  # keep a dependency chain
    mx.eval(x)
for name in sys.argv[1:] or ["stock", "fused"]:
    fn = getattr(mk, name + "_moe")
    for _ in range(5): run(fn)
    t = time.perf_counter(); n = 40
    for _ in range(n): run(fn)
    s = (time.perf_counter() - t) / n
    print(f"{name:8s} {s*1e3:6.2f} ms/token  {per_token/s/1e9:6.1f} GB/s  ({per_token/365e9*1e3:.2f} ms at 365 GB/s)")
