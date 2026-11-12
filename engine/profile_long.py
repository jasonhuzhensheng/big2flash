"""Prompt-chunk component costs (L=512 at ~8K context) and decode-step attention cost at ~12K."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
import flash_next as fn
m = fn.FlashNext("/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw")
H = m.cfg.hidden
def tm(name, f, n=5):
    for _ in range(2): mx.eval(f())
    s = time.perf_counter()
    for _ in range(n): mx.eval(f())
    print(f"{name:44s} {(time.perf_counter()-s)/n*1e3:8.2f} ms", flush=True)
# fill caches to ~8K tokens
m.reset()
ids = [1000 + (i * 7919) % 50000 for i in range(8192)]
for s in range(0, 8192, 512):
    mx.eval(m.forward(ids[s:s + 512]))
saved = [dict(st) for st in m.state]
L = 512
x = mx.random.normal((L, H)) * 0.1; h = mx.random.normal((L, 4 * H)) * 0.1
def restore(i):
    m.state[i] = dict(saved[i]); m.pos = 8192
tm("chunk: full forward (512 tok @ 8K)", lambda: (restore_all(), m.forward(ids[:512]))[1]) if False else None
def restore_all():
    for i in range(len(saved)): m.state[i] = dict(saved[i])
    m.pos = 8192
tm("chunk: full forward (512 tok @ 8K)", lambda: (restore_all(), m.forward(ids[:512]))[1], n=3)
tm("chunk: gated_residual x1", lambda: m.gated_residual(h, "layers.0.attn_hyper_connection"))
tm("chunk: linear_attn layer x1", lambda: (restore(0), m.linear_attn(x, 0))[1])
tm("chunk: attention layer x1 (8K ctx)", lambda: (restore(3), m.attention(x, 3))[1])
tm("chunk: moe layer x1", lambda: m.moe(x, 0))
tm("chunk: ple layer", lambda: (restore(1), m.ple(h, ids[:512], 1))[1], n=3)
tm("chunk: lm_head (all 512 rows)", lambda: fn.linear(x, m.q("language_model.lm_head")))
# decode-step attention at ~12K
for s in range(8192, 12288, 512):
    mx.eval(m.forward(ids[(s % 8192):(s % 8192) + 512]))
saved = [dict(st) for st in m.state]
x1 = mx.random.normal((1, H)) * 0.1
def restore12(i):
    m.state[i] = dict(saved[i]); m.pos = 12288
tm("decode: attention layer x1 (12K ctx)", lambda: (restore12(3), m.attention(x1, 3))[1], n=20)
tm("decode: indexer only x1 (12K ctx)", lambda: (restore12(3), m.indexer_mask(x1, 3, 12288))[1], n=20)
