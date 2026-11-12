"""Quantized matmul cost vs number of rows (1 = decode, 2 = MTP verify) for the 27B's layer shapes."""
import time, mlx.core as mx
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
def t(f, n=50):
    for _ in range(3): mx.eval(f())
    s = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - s) / n * 1e6
for O, I in [(17408, 5120), (5120, 17408), (12288, 5120), (248320, 5120)]:
    # 20 copies so the weights don't sit in cache
    W = [mx.quantize(mx.random.normal((O, I)).astype(mx.float16), 64, 4) for _ in range(8 if O < 100000 else 2)]
    mx.eval(W)
    row = []
    for M in (1, 2, 3, 4, 8):
        x = mx.random.normal((1, M, I)).astype(mx.float16)
        us = t(lambda: [mx.quantized_matmul(x, w, s, b, True, 64, 4) for w, s, b in W]) / len(W)
        row.append(f"M={M}: {us:6.0f}us")
    gb = O * I / 2 * 1.125 / 1e9
    print(f"{O}x{I} ({gb*1000:.0f} MB): " + "  ".join(row), f"| 1-row bw {gb/ (float(row[0].split(':')[1][:-2])/1e6):.0f} GB/s")
