"""Compare FlashNext against the transformers 5.22.1 qwen4_exp reference, layer by layer.

Both sides use the same weights (the artifact, dequantized to FP32 for the reference), so any
difference is in the forward-pass math, not in quantization.

  --side mlx   run the engine on a ~2.6K-token prompt (prefill in 512-token chunks, then 8 cached
               decode steps); dump every layer's output, the indexer selections, and logits.
  --side hf    rebuild each decoder layer from transformers on CPU and run it twice: on the engine's
               input for that layer (per-layer error) and on its own running state (end to end).

The 2.6K prompt is past the indexer budget (2048 tokens = 512 blocks), so QSA block selection is
exercised in prefill and decode.
"""
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
MODEL = "/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw"
PREFIX = "language_model.model."
DECODE = 8
KEEP = 16  # logit rows kept in full (the tail, covering the decode steps)

ap = argparse.ArgumentParser()
ap.add_argument("--side", choices=["mlx", "hf"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default=MODEL)
ap.add_argument("--tokens", type=int, default=2600)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def prompt_ids(tok, n):
    """A two-turn chat (its <|im_end|> tokens exercise the eos-aware n-gram shifting) whose first
    user turn is real source text cut to make the prompt n tokens long."""
    import transformers.models.qwen4_exp.modular_qwen4_exp as src
    body = open(src.__file__).read()
    def build(text):
        msgs = [{"role": "user", "content": "Summarise this file:\n" + text},
                {"role": "assistant", "content": "It defines the Qwen4Exp model."},
                {"role": "user", "content": "Which class implements the block indexer, and what is its budget?"}]
        return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False))["input_ids"]
    lo, hi = 0, len(body)
    while hi - lo > 8:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if len(build(body[:mid])) <= n else (lo, mid)
    return build(body[:lo])


# ============================================================ engine side
if args.side == "mlx":
    import mlx.core as mx
    from transformers import AutoTokenizer
    from flash_next import FlashNext

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = np.array(prompt_ids(tok, args.tokens), dtype=np.int64)
    T = len(ids)
    m = FlashNext(args.model)
    c = m.cfg
    print(f"prompt {T} tokens; eos {c.eos} appears {int((ids == c.eos).sum())}x", flush=True)

    # every _inject call is one residual update; the 2nd of each layer is that layer's output
    rec, sel = [], {}
    inject = m._inject
    m._inject = lambda h, o, inj: rec.append(inject(h, o, inj)) or rec[-1]
    imask = m.indexer_mask
    def indexer_mask(x, i, start):
        r = imask(x, i, start)
        if x.shape[0] > 1:
            sel.setdefault(i, []).append((start, r))
        return r
    m.indexer_mask = indexer_mask

    H = c.hc * c.hidden
    hs = np.lib.format.open_memmap(f"{args.out}/mlx_h.npy", "w+", np.float32, (c.layers, T, H))
    emb0 = None
    argmax, nll = [], []
    m.reset()
    for s in range(0, T, 512):
        chunk = ids[s:s + 512]
        rec.clear()
        logits = m.forward(chunk)
        mx.eval(logits, rec)
        for i in range(c.layers):
            hs[i, s:s + len(chunk)] = np.array(rec[2 * i + 1])
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        argmax.append(np.array(mx.argmax(logits, axis=-1)))
        nxt = np.append(ids[s + 1:s + 1 + len(chunk)], -1)[:len(chunk)]
        nll.append(np.array(-mx.take_along_axis(lp, mx.array(np.maximum(nxt, 0))[:, None], axis=-1)[:, 0]))
    hs.flush()
    full_tail = np.array(logits)[-KEEP:]

    # the selections: (L, T) boolean per attention layer, True = may attend (None = everything causal)
    masks = {}
    for i, parts in sel.items():
        M = np.tril(np.ones((T, T), dtype=bool))
        for start, r in parts:
            if r is not None:
                L = r.shape[0]
                M[start:start + L, :r.shape[1]] &= np.array(r)
        masks[f"sel{i}"] = M

    # cached decode: prefill all but DECODE tokens, then feed them one at a time
    m._inject = inject
    m.indexer_mask = imask
    m.reset()
    N = T - DECODE
    for s in range(0, N, 512):
        logits = m.forward(ids[s:min(s + 512, N)], last_only=True)
        mx.eval(logits)
    dec = [np.array(logits[-1])]
    for t in range(N, T):
        dec.append(np.array(m.forward([int(ids[t])])[-1]))
    np.savez(f"{args.out}/mlx.npz", ids=ids, argmax=np.concatenate(argmax), nll=np.concatenate(nll),
             tail=full_tail, dec=np.stack(dec), **masks)
    print("saved", flush=True)
    sys.exit(0)


# ============================================================ reference side
import mlx.core as mx
import torch
from transformers import AutoConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as Q
from flash_next import NGramTable

torch.set_grad_enabled(False)
tc = AutoConfig.from_pretrained(args.model).text_config
tc._attn_implementation = "sdpa"
d = np.load(f"{args.out}/mlx.npz")
ids = torch.from_numpy(d["ids"])[None]
T = ids.shape[1]
hs = np.load(f"{args.out}/mlx_h.npy", mmap_mode="r")
index = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]
shards = {}
NORMS = ("hc_norm", "q_norm", "k_norm", "q_layernorm", "k_layernorm", "norm_key", "norm_query", "norm_conv")


def load(key):
    f = index[key]
    if f not in shards:
        shards.clear()
        shards[f] = mx.load(os.path.join(args.model, f))
    return shards[f][key]


def tensor(name):
    """Artifact tensor `name` (no .weight suffix for quantized ones) -> FP32 torch, HF convention."""
    k = PREFIX + name if not name.startswith("language_model") else name
    if k + ".scales" in index:
        w, s, b = load(k + ".weight"), load(k + ".scales"), load(k + ".biases")
        bits = w.shape[-1] * 32 // (s.shape[-1] * 64)
        # FP32 scales: with the stored bf16 ones mx.dequantize returns bf16-rounded weights
        a = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=bits)
    else:
        a = load(k + ".weight" if k + ".weight" in index else k)
        a = a.astype(mx.float32) if a.dtype not in (mx.int64, mx.int32, mx.uint32) else a.astype(mx.int64)
    t = torch.from_numpy(np.array(a))
    del a
    return t


def layer_state(i, module):
    sd = {}
    for k, v in module.state_dict().items():
        base = k[:-len(".weight")] if k.endswith(".weight") else k
        if k == "mlp.experts.gate_up_proj":
            t = torch.cat([tensor(f"layers.{i}.mlp.switch_mlp.gate_proj"), tensor(f"layers.{i}.mlp.switch_mlp.up_proj")], 1)
        elif k == "mlp.experts.down_proj":
            t = tensor(f"layers.{i}.mlp.switch_mlp.down_proj")
        else:
            t = tensor(f"layers.{i}.{base}")
            if base.endswith("conv1d"):
                t = t.permute(0, 2, 1).contiguous()  # MLX (C, K, 1) -> torch (C, 1, K)
            if base.rsplit(".", 1)[-1] in NORMS:
                t = t - 1.0  # artifact stores 1 + w; Qwen4ExpTextRMSNorm applies (1 + w)
        assert tuple(t.shape) == tuple(v.shape), (k, t.shape, v.shape)
        sd[k] = t.to(v.dtype)
    return sd


class TableEmbedding(torch.nn.Module):
    """Stands in for the 320M-row nn.Embedding: same rows, read from the 4-bit table on disk.
    The n-gram hashing that picks the rows is the reference's own code."""
    def __init__(self, table):
        super().__init__()
        self.table = table
        self.weight = torch.empty(0)  # the reference reads .weight.device for placement
    def forward(self, idx):
        rows = self.table.rows(idx.reshape(-1).numpy())
        return torch.from_numpy(rows).reshape(*idx.shape, -1)


def rel(a, b):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return ((a - b).norm() / b.norm()).item(), (a - b).abs().max().item()


# the n-gram hash constants the engine reads from the artifact vs the reference's own derivation
ple_i = tc.ple_layer_ids[0] - 1
mult = Q._build_layer_multipliers(tc.vocab_size, tc.ngram_size, 0, tc.seed)
sizes, offs, tot = [], [], 0
for h in range((tc.ngram_size - 1) * tc.heads_per_ngram):
    sz = Q._find_nth_prime_after(tc.ngram_vocab_size_base - 1, h + 1)
    sizes.append(sz); offs.append(tot); tot += sz
print("n-gram constants match:",
      torch.equal(mult, tensor(f"layers.{ple_i}.ple.ple_embedding.layer_multipliers")),
      sizes == tensor(f"layers.{ple_i}.ple.ple_embedding.ngram_heads_vocab_sizes").tolist(),
      offs == tensor(f"layers.{ple_i}.ple.ple_embedding.ngram_heads_offsets").tolist(), flush=True)

eos = tc.eos_token_id[0] if isinstance(tc.eos_token_id, list) else tc.eos_token_id
emb = tensor("embed_tokens")[ids[0]][None]
h_ref = emb.repeat(1, 1, tc.hc_count)
rot = Q.Qwen4ExpTextRotaryEmbedding(tc)
pos = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
pe = rot(h_ref, pos)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))[None, None]
table = NGramTable(os.path.join(args.model, "ngram_table.bin"))

# capture the reference indexer's selections
ref_sel = {}
orig_idx_fwd = Q.Qwen4ExpTextQSAIndexer.forward
def idx_fwd(self, *a, **k):
    r = orig_idx_fwd(self, *a, **k)
    ref_sel.setdefault(self.layer_idx, []).append(r[0, 0].clone())
    return r
Q.Qwen4ExpTextQSAIndexer.forward = idx_fwd

print(f"{'layer':>5} {'type':>8} {'per-layer rel':>14} {'max abs':>9} {'end-to-end rel':>15} {'sel':>14}  time", flush=True)
for i in range(tc.num_hidden_layers):
    t0 = time.perf_counter()
    with torch.device("meta"):
        layer = Q.Qwen4ExpTextDecoderLayer(tc, i)
    if layer.ple is not None:
        layer.ple.ple_embedding.ngram_embedding = TableEmbedding(table)
    layer.load_state_dict(layer_state(i, layer), assign=True)
    layer.eval()
    shards.clear(); mx.clear_cache()
    kw = dict(position_embeddings=pe, attention_mask=causal, conv_mask=None, past_key_values=None,
              ple_input_ids=ids)
    x_in = torch.from_numpy(np.array(hs[i - 1]))[None] if i > 0 else h_ref
    ref_sel.clear()
    tf = layer(x_in, **kw)[0]                       # reference on the engine's input
    sel_info = ""
    if i in ref_sel:
        R = (ref_sel[i][0] & causal[0, 0]).numpy()
        Mm = d[f"sel{i}"]
        diff = int((R != Mm).sum())
        sel_info = f"{diff} diffs/{int(R.sum())}"
    h_ref = layer(h_ref, **kw)                      # reference on its own running state
    mine = torch.from_numpy(np.array(hs[i]))
    r1, m1 = rel(mine, tf)
    r2, _ = rel(mine, h_ref[0])
    print(f"{i:5d} {tc.layer_types[i][:8]:>8} {r1:14.2e} {m1:9.2e} {r2:15.2e} {sel_info:>14}  {time.perf_counter() - t0:.0f}s",
          flush=True)
    del layer, tf

# head: hyper-connection mixer + lm_head
with torch.device("meta"):
    mixer = Q.Qwen4ExpTextGatedResidual(tc, use_combine=False)
msd = {"hc_norm.weight": tensor("hyper_connection_mixer.hc_norm") - 1.0,
       "input_mix_weight_down.weight": tensor("hyper_connection_mixer.input_mix_weight_down"),
       "input_mix_weight_up.weight": tensor("hyper_connection_mixer.input_mix_weight_up")}
mixer.load_state_dict(msd, assign=True)
lm = tensor("language_model.lm_head")
logits = torch.cat([mixer(h_ref[:, s:s + 256])[0] @ lm.T for s in range(0, T, 256)])
lp = torch.log_softmax(logits.double(), -1)
nxt = ids[0, 1:]
nll_ref = -lp[:-1].gather(1, nxt[:, None])[:, 0]
nll_mine = torch.from_numpy(d["nll"][:-1]).double()
top_ref = logits.argmax(-1).numpy()

def kl(p_logits, q_logits):
    p = torch.log_softmax(torch.as_tensor(p_logits).double(), -1)
    q = torch.log_softmax(torch.as_tensor(q_logits).double(), -1)
    return (p.exp() * (p - q)).sum(-1)

tail_ref = logits[-KEEP:]
dec_ref = logits[T - DECODE - 1:]
print("\n== logits (reference chained end to end vs engine)")
print(f"top-1 agreement, all {T} positions (prefill): {(top_ref == d['argmax']).mean() * 100:.2f}%")
print(f"perplexity on the prompt: engine {math.exp(nll_mine.mean()):.4f}  reference {math.exp(nll_ref.mean()):.4f}")
print(f"prefill tail rows: rel {rel(d['tail'], tail_ref)[0]:.2e}, max |dlogit| {rel(d['tail'], tail_ref)[1]:.3f}, "
      f"KL mean {kl(tail_ref, d['tail']).mean():.2e}")
print(f"cached decode ({DECODE} steps + prefill tip): rel {rel(d['dec'], dec_ref)[0]:.2e}, "
      f"max |dlogit| {rel(d['dec'], dec_ref)[1]:.3f}, KL max {kl(dec_ref, d['dec']).max():.2e}, "
      f"top-1 {np.mean(d['dec'].argmax(-1) == dec_ref.argmax(-1).numpy()) * 100:.0f}%")
print(f"cached decode vs engine prefill: rel {rel(d['dec'][-KEEP:], d['tail'][-(DECODE + 1):])[0]:.2e}")
