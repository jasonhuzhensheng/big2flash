"""Per-shape config sweep for qmv8 vs stock quantized_matmul (8-bit). Distinct weight copies
per eval (>= ~400 MB) so the 48 MB system cache cannot serve them."""
import os, sys, time, itertools
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
import dense_kernel as dk

SHAPES = [(16480, 2560), (13952, 2560), (2560, 6144), (320, 10240), (10240, 320),
          (1792, 2560), (2560, 640), (248320, 2560)]

def mk(N, K):
    g = (N, K // 64)
    t = (mx.random.randint(0, 1 << 30, (N, K // 4)).astype(mx.uint32),
         (mx.random.uniform(shape=g) * 1e-3).astype(mx.bfloat16), mx.zeros(g, mx.bfloat16))
    mx.eval(*t); return t

def timeit(f, n):
    for _ in range(3): f()
    t = time.perf_counter()
    for _ in range(10): f()
    return (time.perf_counter() - t) / 10 / n

best = {}
for N, K in SHAPES:
    nb1 = N * K * 1.0625
    n = max(2, min(64, int(4e8 / nb1)))
    ws = [mk(N, K) for _ in range(n)]
    x = mx.random.normal((K,)); mx.eval(x)
    t_stock = timeit(lambda: mx.eval([dk.stock_qmv8(x, *w) for w in ws]), n)
    CH = K // 16
    results = []
    for lpr, rb, rows in itertools.product((2, 4, 8, 16, 32), (1, 2, 4, 8), (2, 4, 8)):
        if CH % lpr or CH // lpr > 40 or CH // lpr < 2:
            continue
        cfg = (lpr, rb, rows)
        t = timeit(lambda: mx.eval([dk.qmv8(x, *w, cfg=cfg) for w in ws]), n)
        results.append((t, cfg))
    results.sort()
    t, cfg = results[0]
    best[K] = best.get(K, []) + [(N, cfg, t)]
    print(f"{N:6d} x {K:5d}  {nb1/1e6:7.1f} MB  stock {t_stock*1e6:7.1f} us {nb1/t_stock/1e9:5.0f} GB/s | "
          f"qmv8 best {cfg} {t*1e6:7.1f} us {nb1/t/1e9:5.0f} GB/s | 2nd {results[1][1]} {nb1/results[1][0]/1e9:5.0f} GB/s",
          flush=True)
    del ws
