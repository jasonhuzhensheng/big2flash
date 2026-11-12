"""Time each block of a decode step on real weights (layer 0 linear, layer 3 attention)."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
import flash_next as fn
m = fn.FlashNext("/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw")
mx.eval(m.forward(list(range(1000, 1064))))
def t(name, f, n=30):
    for _ in range(3): mx.eval(f())
    s = time.perf_counter()
    for _ in range(n): mx.eval(f())
    print(f"{name:40s} {(time.perf_counter()-s)/n*1e3:7.2f} ms", flush=True)
H = m.cfg.hidden
h = mx.random.normal((1, 4 * H)) * 0.1; x = mx.random.normal((1, H)) * 0.1
for dt in (mx.float32, mx.bfloat16):
    q = m.q("layers.0.linear_attn.in_proj_qkv")
    t(f"qmv 10240x2560 8-bit x={dt}", lambda: fn.linear(x.astype(dt), q))
    gw = m.q("layers.0.mlp.switch_mlp.gate_proj")
    idx = mx.array([[1, 5, 9, 33, 70, 100, 200, 300, 400, 500]], dtype=mx.uint32)
    t(f"gather_qmm 10 experts x={dt}", lambda: mx.gather_qmm(x.astype(dt).reshape(1, 1, 1, H), *gw, rhs_indices=idx, transpose=True, group_size=64, bits=4))
t("gated_residual", lambda: m.gated_residual(h, "layers.0.attn_hyper_connection"))
saved = [dict(s) for s in m.state]
def lin():
    m.state[0] = dict(saved[0]); return m.linear_attn(x, 0)
t("linear_attn layer", lin)
def att():
    m.state[3] = dict(saved[3]); return m.attention(x, 3)
t("attention layer (ctx 69)", att)
t("moe layer", lambda: m.moe(x, 0))
t("lm_head", lambda: fn.linear(x, m.q("language_model.lm_head")))
def ple():
    m.state[1] = dict(saved[1]); return m.ple(h, [4242], 1)
t("ple layer (incl. n-gram reads)", ple, n=10)
