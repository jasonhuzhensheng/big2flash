"""engine/qmm8 vs stock quantized_matmul: correctness, then a dependent MLP chain at T=1..8, and a cfg sweep."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
import mlx.core as mx
from qmm8 import qmm8
H, F, L = 5120, 17408, 12
up = [mx.quantize(mx.random.normal((F, H), dtype=mx.float16) * 0.02, group_size=64, bits=4) for _ in range(L)]
dn = [mx.quantize(mx.random.normal((H, F), dtype=mx.float16) * 0.02, group_size=64, bits=4) for _ in range(L)]
mx.eval(up, dn)
nb = sum(a.nbytes for w in up + dn for a in w)
stock = lambda h, w: mx.quantized_matmul(h, *w, transpose=True, group_size=64, bits=4)
for (W, K) in ((up[0], H), (dn[0], F)):
    for T in (2, 5, 8):
        xi = mx.random.normal((T, K), dtype=mx.float16)
        ref, y = stock(xi, W).astype(mx.float32), qmm8(xi, *W).astype(mx.float32)
        print(f"check N={W[0].shape[0]} T={T}: max err {mx.abs(y-ref).max().item():.4f} of max {mx.abs(ref).max().item():.2f}")
def timeit(f):
    f(); best = 1e9
    for _ in range(5):
        t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
    return best * 1000
def chain(x, fu, fd):
    h = x
    for u, d in zip(up, dn):
        h = fd(fu(h, u) * 0.01, d) * 0.01
    mx.eval(h)
if "--sweep" in sys.argv:
    x = mx.random.normal((4, H), dtype=mx.float16)
    for cu in ((1, 1), (2, 1), (4, 1), (1, 2), (2, 2)):
        for cd in ((1, 1), (1, 2), (1, 4), (2, 4), (1, 8)):
            print(cu, cd, f"{timeit(lambda: chain(x, lambda h, w: qmm8(h, *w, cfg=cu), lambda h, w: qmm8(h, *w, cfg=cd))):.1f}")
for T in (1, 2, 3, 4, 6, 8):
    x = mx.random.normal((T, H), dtype=mx.float16)
    ts = timeit(lambda: chain(x, stock, stock))
    tq = timeit(lambda: chain(x, lambda h, w: qmm8(h, *w), lambda h, w: qmm8(h, *w))) if T > 1 else float("nan")
    print(f"T={T} stock {ts:5.1f} ms ({nb/ts/1e6:4.0f} GB/s)  qmm8 {tq:5.1f} ms ({nb/tq/1e6:4.0f} GB/s)", flush=True)
