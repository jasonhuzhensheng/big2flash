"""qmv8 against an FP32 oracle on real Flash-Next dense tensors (every row length the model uses)."""
import json, os, sys
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
from dense_kernel import oracle_qmv8, qmv8, stock_qmv8

MODEL = sys.argv[1]
wmap = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
P = "language_model.model.layers."
names = [
    P + "0.linear_attn.in_proj_qkv",                  # 10240 x 2560
    P + "0.linear_attn.out_proj",                     # 2560 x 6144
    P + "0.attn_hyper_connection.input_mix_weight_down",  # 320 x 10240
    P + "0.attn_hyper_connection.input_mix_weight_up",    # 10240 x 320
    P + "0.mlp.shared_expert.down_proj",              # 2560 x 640
    P + "0.mlp.gate",                                 # 512 x 2560 (N not a multiple of a block)
    P + "0.linear_attn.in_proj_a",                    # 48 x 2560
    "language_model.lm_head",                         # 248320 x 2560
]
worst = 0.0
for n in names:
    w, s, b = (mx.load(os.path.join(MODEL, wmap[f"{n}.{t}"]))[f"{n}.{t}"] for t in ("weight", "scales", "biases"))
    for seed in range(3):
        mx.random.seed(seed)
        x = mx.random.normal((w.shape[1] * 4,))
        ref = oracle_qmv8(x, w, s, b)
        e_f = (mx.linalg.norm(qmv8(x, w, s, b) - ref) / mx.linalg.norm(ref)).item()
        e_s = (mx.linalg.norm(stock_qmv8(x, w, s, b).astype(mx.float32) - ref) / mx.linalg.norm(ref)).item()
        worst = max(worst, e_f)
    print(f"{n.split('.', 2)[-1]:48s} {w.shape[0]:6d} x {w.shape[1] * 4:5d}  fused {e_f:.1e}  stock(bf16) {e_s:.1e}")
print(f"worst fused relative L2 error: {worst:.2e}")
