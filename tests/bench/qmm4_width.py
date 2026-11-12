"""qmm4 (weights read once for all T rows) vs stock quantized_matmul, T=1..16, MLP shapes, dependent chain.
Sweeps the kernel's (LPR, RB, ROWS) per (shape, T) and checks the result against stock."""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx
from qmm4_experiment import qmm4
H, F, L = 5120, 17408, 12
up = [mx.quantize(mx.random.normal((F, H), dtype=mx.float16) * 0.02, group_size=64, bits=4) for _ in range(L)]
dn = [mx.quantize(mx.random.normal((H, F), dtype=mx.float16) * 0.02, group_size=64, bits=4) for _ in range(L)]
mx.eval(up, dn)
nb = sum(a.nbytes for w in up + dn for a in w)

def chain(x, f_up, f_dn):
    h = x
    for u, d in zip(up, dn):
        h = f_dn(f_up(h, u) * 0.01, d) * 0.01
    return h

stock = lambda h, w: mx.quantized_matmul(h, *w, transpose=True, group_size=64, bits=4)
def timeit(f):
    f(); best = 1e9
    for _ in range(4):
        t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
    return best * 1000

cfgs = [c for c in itertools.product((2, 4, 8), (1, 2, 4), (1, 2, 4))]
for T in (1, 2, 3, 4, 6, 8, 12, 16):
    x = mx.random.normal((T, H), dtype=mx.float16)
    ts = timeit(lambda: mx.eval(chain(x, stock, stock)))
    res = {}
    for which, (N, K) in (("up", (F, H)), ("dn", (H, F))):
        best = None
        for c in cfgs:
            lpr, rb, rows = c
            if (K // 32) % lpr: continue
            if T * rb > 32: continue
            W = up[0] if which == "up" else dn[0]
            xi = mx.random.normal((T, K), dtype=mx.float16)
            try:
                y = qmm4(xi, *W, cfg=c); mx.eval(y)
            except Exception:
                continue
            ref = stock(xi, W)
            err = mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item()
            if err > 0.05 * (mx.abs(ref).max().item() + 1e-3): continue
            ws = up if which == "up" else dn
            def f():
                ys = [qmm4(xi, *w, cfg=c) for w in ws]; mx.eval(ys)
            t = timeit(f)
            if best is None or t < best[0]: best = (t, c)
        res[which] = best
    c_up, c_dn = res["up"][1], res["dn"][1]
    tq = timeit(lambda: mx.eval(chain(x, lambda h, w: qmm4(h, *w, cfg=c_up), lambda h, w: qmm4(h, *w, cfg=c_dn))))
    print(f"T={T:2d} stock {ts:6.1f} ms ({nb/ts/1e6:4.0f} GB/s)   qmm4 {tq:6.1f} ms ({nb/tq/1e6:4.0f} GB/s)  up{c_up} dn{c_dn}", flush=True)
