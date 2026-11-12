"""Prompt-chunk (L=512) op costs: dense qmm FP32 vs bf16; expert gather unsorted vs sorted."""
import time
import mlx.core as mx
def mk(N, K, bits, E=None):
    sh = (N, K * bits // 32) if E is None else (E, N, K * bits // 32)
    g = sh[:-1] + (K // 64,)
    t = (mx.random.randint(0, 1 << 30, sh).astype(mx.uint32), (mx.random.uniform(shape=g) * 1e-3).astype(mx.bfloat16),
         mx.zeros(g, mx.bfloat16)); mx.eval(*t); return t
def tm(f, n=10):
    for _ in range(2): mx.eval(f())
    s = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - s) / n * 1e3
L = 512
w = mk(10240, 2560, 8)
for dt in (mx.float32, mx.bfloat16):
    x = mx.random.normal((L, 2560)).astype(dt)
    ms = tm(lambda: mx.quantized_matmul(x, *w, transpose=True, group_size=64, bits=8))
    print(f"dense qmm 512x2560 @ 10240x2560 8-bit, x {dt}: {ms:6.2f} ms ({2*L*10240*2560/ms/1e9:5.2f} TFLOP/s)")
ew = mk(640, 2560, 4, 512)
idx = mx.argpartition(-mx.random.uniform(shape=(L, 512)), 9, axis=-1)[:, :10].astype(mx.uint32)
mx.eval(idx)
for dt in (mx.float32, mx.bfloat16):
    x = mx.random.normal((L, 1, 1, 2560)).astype(dt)
    ms = tm(lambda: mx.gather_qmm(x, *ew, rhs_indices=idx, transpose=True, group_size=64, bits=4))
    print(f"gather_qmm unsorted, x {dt}: {ms:6.2f} ms")
    flat = idx.flatten(); order = mx.argsort(flat); inv = mx.argsort(order)
    xs = mx.random.normal((L * 10, 1, 2560)).astype(dt)
    ms = tm(lambda: mx.gather_qmm(xs, *ew, rhs_indices=flat[order], transpose=True, group_size=64, bits=4, sorted_indices=True))
    print(f"gather_qmm sorted,   x {dt}: {ms:6.2f} ms  ({2*L*10*640*2560/ms/1e9:5.2f} TFLOP/s)")
