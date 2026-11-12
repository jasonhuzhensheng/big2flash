"""Per-kernel efficiency of MLX's quantized matvec and expert gather at Flash-Next shapes."""
import time
import mlx.core as mx

GS = 64


def q(out, inp, bits, n=None):
    shape = (out, inp * bits // 32) if n is None else (n, out, inp * bits // 32)
    g = shape[:-1] + (inp // GS,)
    t = (mx.random.randint(0, 1 << 30, shape).astype(mx.uint32),
         (mx.random.uniform(shape=g) * 1e-3).astype(mx.float16), mx.zeros(g, mx.float16))
    mx.eval(*t)
    return t


def run(name, make_out, nbytes, reps=48, iters=20):
    # reps independent outputs per eval so per-eval overhead amortises; distinct weights defeat the cache
    for _ in range(3):
        mx.eval(make_out())
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(make_out())
    s = (time.perf_counter() - t) / iters
    print(f"{name:34s} {s / reps * 1e6:8.1f} us/call  {nbytes / (s / reps) / 1e9:6.1f} GB/s")


x = mx.random.normal((1, 2560)).astype(mx.float16)
for (o, i, b) in [(10240, 2560, 8), (6144, 2560, 8), (2560, 6144, 8), (320, 10240, 8), (640, 2560, 8), (248320, 2560, 8)]:
    n = 48 if o * i < 1e8 else 4
    ws = [q(o, i, b) for _ in range(n)]
    xi = mx.random.normal((1, i)).astype(mx.float16)
    nb = sum(a.nbytes for a in ws[0])
    run(f"qmv {o}x{i} {b}-bit", lambda: [mx.quantized_matmul(xi, *w, transpose=True, group_size=GS, bits=b) for w in ws], nb, reps=n)

banks = [q(640, 2560, 4, 512) for _ in range(8)]
eb = sum(a.nbytes for a in banks[0]) / 512
xe = mx.random.normal((1, 1, 1, 2560)).astype(mx.float16)
def experts():
    outs = []
    for l in range(48):
        idx = mx.argpartition(-mx.random.uniform(shape=(1, 512)), 9, axis=-1)[..., :10]
        outs.append(mx.gather_qmm(xe, *banks[l % 8], rhs_indices=idx, transpose=True, group_size=GS, bits=4))
    return outs
run("gather_qmm 10 of 512, 640x2560 4b", experts, 10 * eb)
