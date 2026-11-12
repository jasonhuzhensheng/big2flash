"""M2 Max roofline probes for a Qwen3.8-Flash-Next decode step, using stock MLX ops.

1. raw read bandwidth; 2. per-dispatch overhead; 3. a synthetic decode step with the
artifact's real shapes and bit widths (dense 8-bit, routed experts 4-bit, group 64),
random weights, GPU-side top-10 routing. Reports ms/token and effective GB/s.
"""
import time
import mlx.core as mx

H, GS = 2560, 64
mx.random.seed(0)


def bench(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters


# 1. read bandwidth
big = mx.random.uniform(shape=(1 << 29,))  # 2 GiB fp32
mx.eval(big)
s = bench(lambda: mx.eval(mx.sum(big)), 20)
print(f"[bandwidth] sum over 2 GiB: {big.nbytes / s / 1e9:6.1f} GB/s")
del big

# 2. dispatch overhead
v = mx.ones((H,))
def chain(n):
    x = v
    for _ in range(n):
        x = x * 1.0001 + 1.0
    mx.eval(x)
t1, t2 = bench(lambda: chain(100), 20), bench(lambda: chain(1100), 20)
print(f"[dispatch] {(t2 - t1) / 2000 * 1e6:5.2f} us per tiny kernel")


# 3. synthetic decode step
def qweight(out, inp, bits, n=None):
    shape = (out, inp * bits // 32) if n is None else (n, out, inp * bits // 32)
    gshape = shape[:-1] + (inp // GS,)
    w = mx.random.randint(0, 1 << 30, shape).astype(mx.uint32)
    sc = (mx.random.uniform(shape=gshape) * 1e-3).astype(mx.float16)
    bi = mx.zeros(gshape, mx.float16)
    mx.eval(w, sc, bi)
    return (w, sc, bi, bits)


def qmv(x, q):
    w, sc, bi, bits = q
    return mx.quantized_matmul(x, w, sc, bi, transpose=True, group_size=GS, bits=bits)


def nbytes(q):
    return sum(a.nbytes for a in q[:3])


L, E, K, BANKS = 48, 512, 10, 8
layers, total = [], 0
for l in range(L):
    full = (l % 4) == 3
    d = {}
    if full:
        d["q"], d["k"], d["v"] = qweight(12288, H, 8), qweight(512, H, 8), qweight(512, H, 8)
        d["o"], d["idx"] = qweight(H, 6144, 8), qweight(640, H, 8)
    else:
        d["qkv"], d["z"] = qweight(10240, H, 8), qweight(6144, H, 8)
        d["ab"], d["o"] = qweight(128, H, 8), qweight(H, 6144, 8)
    for p in ("a", "m"):
        d[p + "hd"], d[p + "hu"] = qweight(320, 4 * H, 8), qweight(4 * H, 320, 8)
    d["router"] = qweight(E, H, 8)
    d["sg"], d["su"], d["sd"] = qweight(640, H, 8), qweight(640, H, 8), qweight(H, 640, 8)
    total += sum(nbytes(q) for q in d.values())
    layers.append(d)
head = qweight(248320, H, 8)
total += nbytes(head)
banks = [dict(g=qweight(640, H, 4, E), u=qweight(640, H, 4, E), d=qweight(H, 640, 4, E)) for _ in range(BANKS)]
expert_each = sum(nbytes(q) for q in banks[0].values()) / E
total += L * K * expert_each
print(f"[model] synthetic step reads {total / 1e9:.3f} GB/token "
      f"(dense {(total - L * K * expert_each) / 1e9:.3f}, experts {L * K * expert_each / 1e9:.3f})")


def gqmm(x, q, idx):
    w, sc, bi, bits = q
    return mx.gather_qmm(x, w, sc, bi, rhs_indices=idx, transpose=True, group_size=GS, bits=bits)


def step(x):
    hc = mx.concatenate([x] * 4, axis=-1)  # 4 hyper-connection streams
    for l, d in enumerate(layers):
        h = qmv(qmv(hc, d["ahd"]), d["ahu"])[..., :H]
        if "q" in d:
            a = qmv(h, d["q"])[..., :6144] + qmv(h, d["k"]).sum() + qmv(h, d["v"]).sum() + qmv(h, d["idx"]).sum()
        else:
            a = qmv(h, d["qkv"])[..., :6144] * mx.sigmoid(qmv(h, d["z"])) + qmv(h, d["ab"]).sum()
        x = x + qmv(a, d["o"])
        h = qmv(qmv(mx.concatenate([x] * 4, axis=-1), d["mhd"]), d["mhu"])[..., :H]
        r = qmv(h, d["router"])
        idx = mx.argpartition(-r, K - 1, axis=-1)[..., :K]
        gate = mx.softmax(mx.take_along_axis(r, idx, axis=-1), axis=-1)
        b = banks[l % BANKS]
        xe = mx.expand_dims(h, (-2, -3))
        y = gqmm(nn_silu(gqmm(xe, b["g"], idx)) * gqmm(xe, b["u"], idx), b["d"], idx)
        y = (y.squeeze(-2) * gate[..., None]).sum(-2)
        s = qmv(nn_silu(qmv(h, d["sg"])) * qmv(h, d["su"]), d["sd"])
        x = x + y + s
        hc = mx.concatenate([x] * 4, axis=-1)
    return qmv(x, head).argmax(-1)


def nn_silu(t):
    return t * mx.sigmoid(t)


x0 = (mx.random.normal((1, H)) * 0.01).astype(mx.float16)
eager = bench(lambda: mx.eval(step(x0)), 20)
print(f"[decode eager]    {eager * 1e3:6.2f} ms/token  {1 / eager:6.1f} tok/s  {total / eager / 1e9:6.1f} GB/s")
cstep = mx.compile(step)
comp = bench(lambda: mx.eval(cstep(x0)), 20)
print(f"[decode compiled] {comp * 1e3:6.2f} ms/token  {1 / comp:6.1f} tok/s  {total / comp / 1e9:6.1f} GB/s")
