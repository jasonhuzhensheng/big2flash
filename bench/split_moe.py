"""Where the fused MoE time goes: each kernel alone, 48 calls per eval, distinct banks."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
import moe_kernel as mk
K, E, H, I, R = 10, 512, mk.H, mk.I, mk.ROWS_PER_TG
def p(o, i):
    g = (E, o, i // 64)
    t = (mx.random.randint(0, 1 << 30, (E, o, i // 8)).astype(mx.uint32),
         (mx.random.uniform(shape=g) * 1e-3).astype(mx.bfloat16), mx.zeros(g, mx.bfloat16)); mx.eval(*t); return t
banks = [(p(I, H), p(I, H), p(H, I)) for _ in range(8)]
idx = [mx.random.permutation(E)[:K].astype(mx.uint32) for _ in range(48)]
gates = mx.softmax(mx.random.normal((K,))); x = mx.random.normal((H,)); hh = mx.random.normal((K * I,)) * 1e-3
mx.eval(idx, gates, x, hh)
tm = [("H", H), ("I", I), ("K", K), ("ROWS", R), ("RG", mk.RG), ("RD", mk.RD)]
def gu(l):
    return mk._gate_up(inputs=[x, idx[l], *banks[l % 8][0], *banks[l % 8][1]], template=tm,
        grid=(K * I // (2 * mk.RG) * 32, 1, 1), threadgroup=(32 * R, 1, 1), output_shapes=[(K * I,)], output_dtypes=[mx.float32])[0]
def dn(l):
    return mk._down(inputs=[hh, idx[l], gates, *banks[l % 8][2]], template=tm,
        grid=(H // mk.RD * 32, 1, 1), threadgroup=(32 * R, 1, 1), output_shapes=[(H,)], output_dtypes=[mx.float32])[0]
eb = [sum(a.nbytes for a in banks[0][j]) / E * K for j in range(3)]
for name, fn, nb in (("gate_up", gu, eb[0] + eb[1]), ("down", dn, eb[2])):
    f = lambda: mx.eval([fn(l) for l in range(48)])
    for _ in range(5): f()
    t = time.perf_counter(); n = 40
    for _ in range(n): f()
    s = (time.perf_counter() - t) / n / 48
    print(f"{name:8s} {s*1e6:6.1f} us/call  {nb/s/1e9:6.1f} GB/s  (roofline {nb/365e9*1e6:5.1f} us)")
