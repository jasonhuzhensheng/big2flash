"""All of a decode token's dense matvecs (36 linear-attention layers, 12 full-attention layers, head)
with distinct random weights at the artifact's shapes, as a dependent chain: stock vs qmv8, separate
projections vs row-concatenated groups. Also an 'independent' run (no dependency) to separate
bandwidth from per-kernel latency."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
import dense_kernel as dk

def mk(N, K):
    g = (N, K // 64)
    t = (mx.random.randint(0, 1 << 30, (N, K // 4)).astype(mx.uint32),
         (mx.random.uniform(shape=g) * 1e-3).astype(mx.bfloat16), mx.zeros(g, mx.bfloat16))
    mx.eval(*t); return t

# per layer, in dependency order; inner lists are projections sharing one input
LIN = [[(320, 10240)], [(10240, 320)], [(10240, 2560), (6144, 2560), (48, 2560), (48, 2560)], [(2560, 6144)],
       [(320, 10240)], [(10240, 320)], [(640, 2560), (640, 2560), (512, 2560)], [(2560, 640)]]
FULL = [[(320, 10240)], [(10240, 320)], [(12288, 2560), (512, 2560), (512, 2560), (640, 2560)], [(2560, 6144)],
        [(320, 10240)], [(10240, 320)], [(640, 2560), (640, 2560), (512, 2560)], [(2560, 640)]]
layers = []
for l in range(48):
    spec = FULL if l % 4 == 3 else LIN
    layers.append([[mk(N, K) for N, K in step] for step in spec])
head = mk(248320, 2560)
total = sum(sum(a.nbytes for a in q) for L in layers for st in L for q in st) + sum(a.nbytes for a in head)

def cat(step):  # row-concatenate projections that share an input (done once at load)
    return tuple(mx.concatenate([q[j] for q in step], axis=0) for j in range(3)) if len(step) > 1 else step[0]
fused_layers = [[cat(st) for st in L] for L in layers]
mx.eval(fused_layers)

def chain(fn, fused):
    x = mx.random.normal((10240,)) * 0.01
    for L in (fused_layers if fused else layers):
        for st in L:
            mats = [st] if fused else st
            K = mats[0][0].shape[1] * 4
            xi = x[:K] if x.shape[0] >= K else mx.concatenate([x] * (K // x.shape[0]))
            ys = [fn(xi, *q) for q in mats]
            x = ys[0]
    return fn(x[:2560], *head).astype(mx.float32)

def independent(fn):
    outs = []
    for L in fused_layers:
        for st in L:
            outs.append(fn(mx.zeros((st[0].shape[1] * 4,)), *st))
    outs.append(fn(mx.zeros((2560,)), *head))
    return outs

def t(f, n=10):
    for _ in range(3): mx.eval(f())
    s = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - s) / n

print(f"dense bytes per token: {total/1e9:.3f} GB  (roofline {total/365e9*1e3:.2f} ms at 365 GB/s)")
for name, fn in (("stock", dk.stock_qmv8), ("qmv8", dk.qmv8)):
    for fused in (False, True):
        s = t(lambda: chain(fn, fused))
        print(f"{name:5s} {'grouped ' if fused else 'separate'} chain: {s*1e3:6.2f} ms  {total/s/1e9:5.0f} GB/s", flush=True)
    s = t(lambda: independent(fn))
    print(f"{name:5s} grouped  independent: {s*1e3:6.2f} ms  {total/s/1e9:5.0f} GB/s", flush=True)
