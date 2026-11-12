"""Check qmm4 against mx.quantized_matmul and sweep (LPR, RB, ROWS) per 27B shape, T = 1 and 2.
Each timing runs a chain over enough distinct weights (>1 GB) that nothing is served from cache."""
import sys, os, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
import mlx.core as mx
import qmm4_experiment as qmm4
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

def bench(f, n=20):
    for _ in range(2): mx.eval(f())
    t = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - t) / n

SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (5120, 6144), (6144, 5120), (12288, 5120), (248320, 5120), (1024, 5120)]
best = {}
for N, K in SHAPES:
    nbytes = N * K // 2 * 1.125
    copies = max(1, int(1.2e9 // nbytes))
    W = [mx.quantize(mx.random.normal((N, K)).astype(mx.float16) * 0.02, 64, 4) for _ in range(copies)]
    mx.eval(W)
    for T in (1, 2):
        x = mx.random.normal((T, K)).astype(mx.float16)
        ref = mx.quantized_matmul(x, *W[0], True, 64, 4)
        stock = bench(lambda: [mx.quantized_matmul(x, w, s, b, True, 64, 4) for w, s, b in W]) / copies
        res = []
        for lpr, rb, rows in itertools.product((4, 8, 16, 32), (1, 2, 4, 8), (1, 2, 4)):
            if (K // 32) % lpr:
                continue
            cfg = (lpr, rb, rows)
            out = qmm4.qmm4(x, *W[0], cfg)
            err = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max() / (mx.abs(ref).max() + 1e-6))
            if err > 2e-3:
                print("BAD", N, K, T, cfg, err); continue
            res.append((bench(lambda: [qmm4.qmm4(x, w, s, b, cfg) for w, s, b in W]) / copies, cfg))
        t, cfg = min(res)
        best[(K, T)] = best.get((K, T), []) + [(N, cfg, t)]
        print(f"{N}x{K} T={T}: stock {stock*1e6:6.0f}us {nbytes/stock/1e9:4.0f} GB/s | qmm4 {t*1e6:6.0f}us "
              f"{nbytes/t/1e9:4.0f} GB/s {cfg}", flush=True)
