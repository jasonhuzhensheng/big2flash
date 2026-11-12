"""Correctness on real Flash-Next layer-0 experts, then 48-layer decode timing: fused vs stock."""
import json, os, sys, time
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
from moe_kernel import H, I, fused_moe, oracle_moe, stock_moe

MODEL = sys.argv[1]
K, E = 10, 512
wmap = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]


def real_proj(layer, name):
    pre = f"language_model.model.layers.{layer}.mlp.switch_mlp.{name}."
    out = []
    for s in ("weight", "scales", "biases"):
        out.append(mx.load(os.path.join(MODEL, wmap[pre + s]))[pre + s])
    mx.eval(*out)
    return tuple(out)


def route(seed):
    mx.random.seed(seed)
    idx = mx.random.permutation(E)[:K].astype(mx.uint32)
    gates = mx.softmax(mx.random.normal((K,)))
    x = mx.random.normal((H,)) * 0.5
    return x, idx, gates


# --- correctness on real weights
gate, up, down = real_proj(0, "gate_proj"), real_proj(0, "up_proj"), real_proj(0, "down_proj")
print("correctness vs FP32 oracle (layer 0 real experts, 5 routes):")
worst = {"fused": 0.0, "stock": 0.0}
for seed in range(5):
    x, idx, gates = route(seed)
    ref = oracle_moe(x, idx, gates, gate, up, down)
    for name, fn in (("fused", fused_moe), ("stock", stock_moe)):
        y = fn(x, idx, gates, gate, up, down)
        rel = (mx.linalg.norm(y - ref) / mx.linalg.norm(ref)).item()
        worst[name] = max(worst[name], rel)
print(f"  worst relative L2 error: fused {worst['fused']:.2e}   stock (bf16 activations) {worst['stock']:.2e}")

# --- timing: 48 layers, 8 real-shaped banks cycled (random contents), GPU-side top-K like a router
banks = [gate, up, down] and [(gate, up, down)]
for b in range(7):
    banks.append(tuple(
        (mx.random.randint(0, 1 << 30, w.shape).astype(mx.uint32), s, bi) for (w, s, bi) in (gate, up, down)))
    mx.eval(*[a for p in banks[-1] for a in p])
bank_bytes = sum(a.nbytes for p in banks[0] for a in p) / E
per_token = 48 * K * bank_bytes


def decode(fn):
    x = mx.random.normal((H,)) * 0.5
    for l in range(48):
        logits = mx.random.uniform(shape=(E,))
        idx = mx.argpartition(-logits, K - 1)[:K].astype(mx.uint32)
        gates = mx.softmax(logits[idx])
        x = x + fn(x, idx, gates, *banks[l % 8])
    mx.eval(x)


for name, fn in (("stock", stock_moe), ("fused", fused_moe)):
    for _ in range(5):
        decode(fn)
    t = time.perf_counter()
    n = 30
    for _ in range(n):
        decode(fn)
    s = (time.perf_counter() - t) / n
    print(f"{name}: experts of 48 layers {s * 1e3:6.2f} ms/token  {per_token / s / 1e9:6.1f} GB/s "
          f"(includes router top-K and residual adds)")
