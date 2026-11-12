"""Which part of the qmm8 kernel limits it: run variants with one piece removed, on the 12 up projections."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
import mlx.core as mx
import qmm8
H, F, L = 5120, 17408, 12
up = [mx.quantize(mx.random.normal((F, H), dtype=mx.float16) * 0.02, group_size=64, bits=4) for _ in range(L)]
mx.eval(up)
nb = sum(a.nbytes for w in up for a in w)
xp = qmm8.permute_x(mx.random.normal((4, H), dtype=mx.float16)); mx.eval(xp)
SRC = qmm8._SRC
VARIANTS = {
    "full": SRC,
    "no-dequant": SRC.replace("sc[r] * half(u & 0xf) + bi[r]", "half(u & 0xf)").replace("sc[r] * half((u >> 4) & 0xf) + bi[r]", "half(u >> 4)"),
    "x": SRC.replace("simdgroup_multiply_accumulate(acc[r], wm, xm, acc[r]);", "acc[r].thread_elements()[0] += float(wm.thread_elements()[0] * xm.thread_elements()[0]); acc[r].thread_elements()[1] += float(wm.thread_elements()[1]);"),
    "no-xload": SRC.replace("simdgroup_load(xm, xb + j * 64, 8);", "xm = simdgroup_matrix<half, 8, 8>(half(j));"),
    "loads-only": SRC.replace("simdgroup_multiply_accumulate(acc[r], wm, xm, acc[r]);", "acc[r].thread_elements()[0] += float(u & 0xff) + float(xm.thread_elements()[0]);").replace("simdgroup_load(xm, xb + j * 64, 8);", "simdgroup_load(xm, xb + j * 64, 8);"),
}
for name, src in VARIANTS.items():
    k = mx.fast.metal_kernel(name="ab_" + name.replace("-", "_"), input_names=["xp", "w", "s", "b"], output_names=["y"], source=src)
    for cfg in ((2, 1), (2, 2), (4, 1), (1, 1)):
        R, sp = cfg
        def run():
            ys = [k(inputs=[xp, *w], template=[("N", F), ("K", H), ("R", R), ("SPLIT", sp)],
                    grid=(F // (8 * R) * 32 * sp, 1, 1), threadgroup=(32 * sp, 1, 1),
                    output_shapes=[(8, F)], output_dtypes=[mx.float16])[0] for w in up]
            mx.eval(ys)
        run(); best = 1e9
        for _ in range(5):
            t = time.perf_counter(); run(); best = min(best, time.perf_counter() - t)
        print(f"{name:11s} R={R} split={sp}  {best*1000:5.2f} ms  {nb/best/1e9:4.0f} GB/s", flush=True)
x1 = mx.random.normal((1, H), dtype=mx.float16)
def st():
    mx.eval([mx.quantized_matmul(x1, *w, transpose=True, group_size=64, bits=4) for w in up])
st(); best = 1e9
for _ in range(5):
    t = time.perf_counter(); st(); best = min(best, time.perf_counter() - t)
print(f"stock T=1   {best*1000:5.2f} ms  {nb/best/1e9:4.0f} GB/s")
