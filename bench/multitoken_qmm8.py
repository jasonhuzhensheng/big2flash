"""qmm8 (T tokens per weight load) vs stock: correctness against the FP32 oracle, then T scaling."""
import os, sys, time, itertools
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
import dense_kernel as dk
def mk(N, K):
    g = (N, K // 64)
    t = (mx.random.randint(0, 1 << 30, (N, K // 4)).astype(mx.uint32),
         (mx.random.uniform(shape=g) * 1e-2).astype(mx.bfloat16), (mx.random.normal(g) * 1e-2).astype(mx.bfloat16)); mx.eval(*t); return t
# correctness
w = mk(2560, 6144); x = mx.random.normal((5, 6144))
ref = mx.dequantize(w[0], w[1].astype(mx.float32), w[2].astype(mx.float32), group_size=64, bits=8) @ x.T
err = (mx.linalg.norm(dk.qmm8(x, *w) - ref.T) / mx.linalg.norm(ref)).item()
print(f"qmm8 T=5 relative L2 error vs FP32 oracle: {err:.1e}")
Ts = (1, 2, 4, 6, 8)
for N, K in [(16480, 2560), (2560, 6144), (320, 10240), (10240, 320), (2560, 640)]:
    n = max(2, min(64, int(4e8 / (N * K))))
    ws = [mk(N, K) for _ in range(n)]
    def tm(f):
        for _ in range(3): f()
        s = time.perf_counter()
        for _ in range(10): f()
        return (time.perf_counter() - s) / 10 / n * 1e6
    out = []
    for T in Ts:
        x = mx.random.normal((T, K)); xb = x.astype(mx.bfloat16); mx.eval(x, xb)
        st = tm(lambda: mx.eval([mx.quantized_matmul(xb, *q, transpose=True, group_size=64, bits=8) for q in ws]))
        lpr0 = dk.config(K)[0]
        best = min(tm(lambda: mx.eval([dk.qmm8(x, *q, cfg=(lpr0, rb, rows)) for q in ws])) for rb, rows in
                   itertools.product((1, 2, 4), (2, 4)) if rb * T <= 32)
        out.append(f"T={T}: stock {st:6.1f} / qmm8 {best:6.1f}")
    print(f"{N:6d}x{K:5d}  " + " | ".join(out), flush=True)
    del ws
