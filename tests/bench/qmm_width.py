"""Quantized matmul time vs number of rows M, on the 27B's MLP shapes, in a dependent chain."""
import time, mlx.core as mx
H, F, L = 5120, 17408, 24
ws = []
for _ in range(L):
    for (o, i) in ((F, H), (H, F)):
        ws.append(mx.quantize(mx.random.normal((o, i), dtype=mx.float16) * 0.02, group_size=64, bits=4))
mx.eval(ws)
nb = sum(a.nbytes for w in ws for a in w)
for M in (1, 2, 3, 4, 5, 6, 8, 12, 16, 32):
    x = mx.random.normal((M, H), dtype=mx.float16)
    def run():
        h = x
        for j in range(0, len(ws), 2):
            u = mx.quantized_matmul(h, *ws[j], transpose=True, group_size=64, bits=4)
            h = mx.quantized_matmul(u * 0.01, *ws[j + 1], transpose=True, group_size=64, bits=4) * 0.01
        mx.eval(h)
    run(); best = 1e9
    for _ in range(5):
        t = time.perf_counter(); run(); best = min(best, time.perf_counter() - t)
    print(f"M={M:2d} {best*1000:6.1f} ms  {nb/best/1e9:5.0f} GB/s", flush=True)
