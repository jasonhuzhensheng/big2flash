"""Block-by-block comparison of one decoder layer: FlashNext (MLX) vs transformers (CPU, FP32).

Loads only that layer's weights into a bare FlashNext, so it runs without the full 68 GB model.
Usage: compare_blocks.py [layer] [tokens]
"""
import json, os, sys
import numpy as np
import mlx.core as mx
import torch

sys.path.insert(0, os.path.dirname(__file__))
from flash_next import FlashNext, Config, PREFIX
from transformers import AutoConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as Q

MODEL = "/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw"
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 256
NORMS = ("hc_norm", "q_norm", "k_norm", "q_layernorm", "k_layernorm", "norm_key", "norm_query", "norm_conv")
torch.set_grad_enabled(False)

index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
want = [k for k in index if k.startswith(f"{PREFIX}layers.{LAYER}.")]
w = {}
for f in sorted({index[k] for k in want}):
    data = mx.load(os.path.join(MODEL, f))
    w.update({k: data[k] * 1 for k in want if index[k] == f})
mx.eval(w)

m = FlashNext.__new__(FlashNext)
m.cfg = c = Config.load(MODEL)
m.w = w
m._compile_blocks()
m.reset()


def deq(name):
    q = m.q(f"layers.{LAYER}.{name}")
    if isinstance(q, tuple):
        wq, s, b = q
        return mx.dequantize(wq, s.astype(mx.float32), b.astype(mx.float32), group_size=64,
                             bits=wq.shape[-1] * 32 // (s.shape[-1] * 64))
    return q.astype(mx.float32)


def T(a):
    return torch.from_numpy(np.array(a.astype(mx.float32)))


tc = AutoConfig.from_pretrained(MODEL).text_config
tc._attn_implementation = "sdpa"
with torch.device("meta"):
    layer = Q.Qwen4ExpTextDecoderLayer(tc, LAYER)
sd = {}
for k, v in layer.state_dict().items():
    base = k[:-7] if k.endswith(".weight") else k
    if k == "mlp.experts.gate_up_proj":
        t = torch.cat([T(deq("mlp.switch_mlp.gate_proj")), T(deq("mlp.switch_mlp.up_proj"))], 1)
    elif k == "mlp.experts.down_proj":
        t = T(deq("mlp.switch_mlp.down_proj"))
    elif base.startswith("ple."):
        continue
    else:
        t = T(deq(base))
        if base.endswith("conv1d"):
            t = t.permute(0, 2, 1).contiguous()
        if base.rsplit(".", 1)[-1] in NORMS:
            t = t - 1.0
    sd[k] = t
layer.load_state_dict(sd, assign=True, strict=False)
assert layer.ple is None, "pick a layer without PLE"

if len(sys.argv) > 3:  # a layer-output dump from compare_hf.py --side mlx: the real input to LAYER
    h_np = np.load(sys.argv[3], mmap_mode="r")[LAYER - 1, :TOK].copy() if LAYER > 0 else None
    if h_np is None:
        ids = np.load(os.path.join(os.path.dirname(sys.argv[3]), "mlx.npz"))["ids"][:TOK]
        e = {k: mx.load(os.path.join(MODEL, index[k]))[k] for k in
             (PREFIX + "embed_tokens.weight", PREFIX + "embed_tokens.scales", PREFIX + "embed_tokens.biases")}
        ew, es, eb = (e[PREFIX + "embed_tokens." + n] for n in ("weight", "scales", "biases"))
        t = mx.array(ids)
        emb = mx.dequantize(ew[t], es[t].astype(mx.float32), eb[t].astype(mx.float32), group_size=64,
                            bits=ew.shape[-1] * 32 // (es.shape[-1] * 64))
        h_np = np.tile(np.array(emb), (1, c.hc)).astype(np.float32)
else:
    rng = np.random.default_rng(0)
    h_np = (rng.standard_normal((TOK, c.hc * c.hidden)) * 0.05).astype(np.float32)
h = mx.array(h_np)
ht = torch.from_numpy(h_np)[None]


def rel(a, b):
    a = torch.as_tensor(np.array(a) if isinstance(a, mx.array) else a).double().reshape(-1)
    b = torch.as_tensor(b).double().reshape(-1)
    return f"rel {((a - b).norm() / b.norm()).item():.2e}  max {(a - b).abs().max().item():.2e}"


# 1. gated residual (attention side)
x, inj = m.gated_residual(h, f"layers.{LAYER}.attn_hyper_connection")
xr, _, injr = layer.attn_hyper_connection(ht)
print("attn hyper-connection   x:", rel(x, xr[0]), "| inject:", rel(inj, injr[0]))

# 2. mixer block on identical input
xt = torch.from_numpy(np.array(x))[None]
if c.layer_types[LAYER] == "linear_attention":
    o = m.linear_attn(x, LAYER)
    orr = layer.linear_attn(xt, cache_params=None, attention_mask=None)
    print("linear attention        :", rel(o, orr[0]))
else:
    rot = Q.Qwen4ExpTextRotaryEmbedding(tc)
    pe = rot(xt, torch.arange(TOK).view(1, 1, -1).expand(3, 1, -1))
    o = m.attention(x, LAYER)
    orr = layer.self_attn(xt, pe, attention_mask=torch.tril(torch.ones(TOK, TOK, dtype=torch.bool))[None, None])[0]
    print("attention               :", rel(o, orr[0]))

# 3. MoE on identical input (use the reference's mixed input so only the MoE differs)
xm = x
xmt = xt
idx, sc = m._route(xm, m.q(f"layers.{LAYER}.mlp.gate"))
logits_r = torch.nn.functional.linear(xmt[0], layer.mlp.gate.weight)
probs_r = torch.softmax(logits_r, -1, dtype=torch.float)
sc_r, idx_r = probs_r.topk(c.top_k, -1)
same = np.mean([set(a) == set(b) for a, b in zip(np.array(idx).tolist(), idx_r.tolist())])
print(f"router                  : same expert set {same * 100:.1f}% of tokens")
for t in [t for t, (a, b) in enumerate(zip(np.array(idx).tolist(), idx_r.tolist())) if set(a) != set(b)][:5]:
    p = np.sort(probs_r[t].numpy())[::-1]
    print(f"  token {t}: 10th vs 11th expert prob {p[9]:.6f} vs {p[10]:.6f} (gap {p[9] - p[10]:.1e})")
mo = m.moe(xm, LAYER)
mor = layer.mlp(xmt)
print("MoE (routed + shared)   :", rel(mo, mor[0]))
ok = [t for t, (a, b) in enumerate(zip(np.array(idx).tolist(), idx_r.tolist())) if set(a) == set(b)]
print("  tokens with same experts:", rel(np.array(mo)[ok], mor[0].numpy()[ok]))
y_sh = m._shared(xm, *(m.q(f"layers.{LAYER}.mlp.shared_expert.{n}") for n in ("gate_proj", "up_proj", "down_proj")),
                 m.q(f"layers.{LAYER}.mlp.shared_expert_gate"))
y_sh_r = torch.sigmoid(layer.mlp.shared_expert_gate(xmt)) * layer.mlp.shared_expert(xmt)
print("  shared expert         :", rel(y_sh, y_sh_r[0]))
print("  routed experts        :", rel(m.experts(xm, idx, sc, f"layers.{LAYER}.mlp."), (mor - y_sh_r)[0]))

# 4. plain quantized matmul vs dequantized FP32 matmul
qd = m.q(f"layers.{LAYER}.mlp.shared_expert.gate_proj")
print("quantized_matmul vs x@W^T:", rel(
      mx.quantized_matmul(xm, *qd, transpose=True, group_size=64, bits=8), xmt[0] @ T(deq("mlp.shared_expert.gate_proj")).T))
