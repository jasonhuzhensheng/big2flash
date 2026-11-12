"""Dependent chain like an MLP stack: x -> up (17408x5120) -> down (5120x17408) -> ... 32 pairs (2.7 GB)."""
import sys, os, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
import mlx.core as mx
import qmm4_experiment as qmm4
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
L = 32
UP = [mx.quantize(mx.random.normal((17408, 5120)).astype(mx.float16) * 0.02, 64, 4) for _ in range(L)]
DN = [mx.quantize(mx.random.normal((5120, 17408)).astype(mx.float16) * 0.01, 64, 4) for _ in range(L)]
mx.eval(UP, DN)
nbytes = L * 2 * 17408 * 5120 // 2 * 1.125
def run(T, f_up, f_dn):
    x = mx.random.normal((T, 5120)).astype(mx.float16)
    for u, d in zip(UP, DN):
        x = f_dn(f_up(x, u), d)
    return x
def bench(T, f_up, f_dn, n=10):
    mx.eval(run(T, f_up, f_dn))
    t = time.perf_counter()
    for _ in range(n): mx.eval(run(T, f_up, f_dn))
    return (time.perf_counter() - t) / n
stock = lambda x, w: mx.quantized_matmul(x, *w, True, 64, 4)
for T in (2,):
    s = bench(T, stock, stock)
    print(f"T={T} stock {s*1000:.1f} ms {nbytes/s/1e9:.0f} GB/s", flush=True)
    res = []
    for cu in itertools.product((4, 8, 16), (1, 2, 4, 8), (1, 2, 4)):
        for cd in [(8, 2, 2), (8, 4, 2), (16, 2, 2), (16, 4, 2), (32, 2, 2), (32, 4, 2), (16, 1, 4), (32, 1, 4)]:
            pass
    # sweep up and down separately (other side fixed to stock)
    for name, side in (("up", 0), ("down", 1)):
        res = []
        for cfg in itertools.product((4, 8, 16, 32), (1, 2, 4, 8), (1, 2, 4, 8)):
            K = 5120 if side == 0 else 17408
            if (K // 32) % cfg[0]: continue
            f = lambda x, w, c=cfg: qmm4.qmm4(x, *w, c)
            t = bench(T, f, stock, 4) if side == 0 else bench(T, stock, f, 4)
            res.append((t, cfg))
        res.sort()
        print(f"  T={T} {name}: best {res[0][1]} chain {res[0][0]*1000:.1f} ms; top3 {[c for _, c in res[:3]]}", flush=True)
