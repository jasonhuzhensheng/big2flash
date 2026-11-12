"""Sweep block sizes for the fused MoE kernels (kernel-only, 48 calls per eval)."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
import moe_kernel as mk
K, E, H, I = 10, 512, mk.H, mk.I
def p(o, i):
    g = (E, o, i // 64)
    t = (mx.random.randint(0, 1 << 30, (E, o, i // 8)).astype(mx.uint32),
         (mx.random.uniform(shape=g) * 1e-3).astype(mx.bfloat16), mx.zeros(g, mx.bfloat16)); mx.eval(*t); return t
banks = [(p(I, H), p(I, H), p(H, I)) for _ in range(8)]
idx = [mx.random.permutation(E)[:K].astype(mx.uint32) for _ in range(48)]
gates = mx.softmax(mx.random.normal((K,))); x = mx.random.normal((H,)); hh = mx.random.normal((K * I,)) * 1e-3
mx.eval(idx, gates, x, hh)
eb = [sum(a.nbytes for a in banks[0][j]) / E * K for j in range(3)]
def t(f):
    for _ in range(4): f()
    s = time.perf_counter(); n = 30
    for _ in range(n): f()
    return (time.perf_counter() - s) / n / 48
for rows in (4, 8, 16):
    for rg in (1, 2, 4, 8):
        tm = [("H", H), ("I", I), ("K", K), ("ROWS", rows), ("RG", rg), ("RD", 1)]
        f = lambda: mx.eval([mk._gate_up(inputs=[x, idx[l], *banks[l % 8][0], *banks[l % 8][1]], template=tm,
            grid=(K * I // (2 * rg) * 32, 1, 1), threadgroup=(32 * rows, 1, 1), output_shapes=[(K * I,)], output_dtypes=[mx.float32])[0] for l in range(48)])
        s = t(f); print(f"gate_up ROWS={rows:2d} RG={rg}: {s*1e6:6.1f} us  {(eb[0]+eb[1])/s/1e9:6.1f} GB/s", flush=True)
    for rd in (2, 4, 8, 16):
        tm = [("H", H), ("I", I), ("K", K), ("ROWS", rows), ("RG", 1), ("RD", rd)]
        f = lambda: mx.eval([mk._down(inputs=[hh, idx[l], gates, *banks[l % 8][2]], template=tm,
            grid=(H // rd * 32, 1, 1), threadgroup=(32 * rows, 1, 1), output_shapes=[(H,)], output_dtypes=[mx.float32])[0] for l in range(48)])
        s = t(f); print(f"down    ROWS={rows:2d} RD={rd:2d}: {s*1e6:6.1f} us  {eb[2]/s/1e9:6.1f} GB/s", flush=True)
