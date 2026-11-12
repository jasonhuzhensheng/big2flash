"""Cost of verifying T tokens at once: stock 8-bit quantized_matmul at Flash-Next dense shapes."""
import time
import mlx.core as mx
def mk(N, K):
    g = (N, K // 64)
    t = (mx.random.randint(0, 1 << 30, (N, K // 4)).astype(mx.uint32),
         (mx.random.uniform(shape=g) * 1e-3).astype(mx.bfloat16), mx.zeros(g, mx.bfloat16)); mx.eval(*t); return t
for N, K in [(16480, 2560), (2560, 6144), (320, 10240), (10240, 320), (2560, 640)]:
    n = max(2, min(64, int(4e8 / (N * K))))
    ws = [mk(N, K) for _ in range(n)]
    row = []
    for T in (1, 2, 3, 4, 6, 8, 16):
        x = mx.random.normal((T, K)).astype(mx.bfloat16); mx.eval(x)
        f = lambda: mx.eval([mx.quantized_matmul(x, *w, transpose=True, group_size=64, bits=8) for w in ws])
        for _ in range(3): f()
        s = time.perf_counter()
        for _ in range(10): f()
        row.append((time.perf_counter() - s) / 10 / n * 1e6)
    print(f"{N:6d}x{K:5d}  " + "  ".join(f"T={T}:{u:6.1f}us" for T, u in zip((1, 2, 3, 4, 6, 8, 16), row)), flush=True)
    del ws
