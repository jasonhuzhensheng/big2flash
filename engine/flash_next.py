"""Qwen3.8-Flash-Next (`qwen4_exp`) forward pass in MLX, single sequence, FP32 activations.

Follows transformers 5.22.1 `modular_qwen4_exp.py`: 4 hyper-connection streams around every block,
GatedDeltaNet linear attention (36 layers), gated softmax attention with a QSA block indexer
(12 layers), 512-expert top-10 MoE with a gated shared expert, and one PLE layer whose hashed
n-gram embedding table (32 GB, 4-bit) is read from disk through a memmap.

Weights are the MLX-Serve artifact as stored (8-bit dense, 4-bit experts, group 64). RMSNorm
weights there are already offset (+1), so they are applied directly.
"""
import json
import math
import os
from dataclasses import dataclass

from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

from moe_kernel import fused_moe

PREFIX = "language_model.model."
KV_STEP = 4096


@dataclass
class Config:
    hidden: int
    layers: int
    layer_types: list
    heads: int
    kv_heads: int
    head_dim: int
    rope_dims: int
    rope_theta: float
    eps: float
    hc: int
    lin_k_heads: int
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_kernel: int
    experts: int
    top_k: int
    idx_heads: int
    idx_dim: int
    idx_budget: int
    idx_ratio: int
    ple_layers: list
    ngram_size: int
    heads_per_ngram: int
    ple_conv_kernel: int
    eos: int
    vocab: int

    @classmethod
    def load(cls, path):
        c = json.load(open(os.path.join(path, "config.json")))
        t = c.get("text_config", c)
        rp = t["rope_parameters"]
        return cls(
            hidden=t["hidden_size"], layers=t["num_hidden_layers"], layer_types=t["layer_types"],
            heads=t["num_attention_heads"], kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"],
            rope_dims=int(t["head_dim"] * rp["partial_rotary_factor"]), rope_theta=float(rp["rope_theta"]),
            eps=t["rms_norm_eps"], hc=t["hc_count"],
            lin_k_heads=t["linear_num_key_heads"], lin_v_heads=t["linear_num_value_heads"],
            lin_k_dim=t["linear_key_head_dim"], lin_v_dim=t["linear_value_head_dim"],
            conv_kernel=t["linear_conv_kernel_dim"], experts=t["num_experts"], top_k=t["num_experts_per_tok"],
            idx_heads=t["indexer_n_heads"], idx_dim=t["indexer_head_dim"], idx_budget=t["indexer_budget"],
            idx_ratio=t["indexer_compress_ratio"], ple_layers=[i - 1 for i in t.get("ple_layer_ids") or []],
            ngram_size=t["ngram_size"], heads_per_ngram=t["heads_per_ngram"],
            ple_conv_kernel=t["ple_conv_kernel_size"], eos=t["eos_token_id"], vocab=t["vocab_size"])


# ---------------------------------------------------------------- primitives

def qbits(w, s):
    return w.shape[-1] * 32 // (s.shape[-1] * 64)


def linear(x, q):
    """x (..., K) @ quantized (N, K)^T; q = (w, scales, biases) or a plain matrix."""
    if not isinstance(q, tuple):
        return x @ q.astype(mx.float32).T
    w, s, b = q
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=qbits(w, s))


def rms_norm(x, weight, eps, group=None):
    """Qwen4ExpTextRMSNorm with the stored (+1) weight; `group` normalises each slice of that size."""
    shape = x.shape
    if group is not None:
        x = x.reshape(*shape[:-1], -1, group)
    x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x.reshape(shape) * weight.astype(mx.float32)


def silu(x):
    return x * mx.sigmoid(x)


def rope(x, offset, dims, base):
    """x (L, H, D): rotate the first `dims` of D (non-interleaved halves) at positions offset..offset+L-1.
    Text-only multimodal RoPE has equal t/h/w positions, which is plain RoPE."""
    return mx.fast.rope(x.transpose(1, 0, 2)[None], dims, traditional=False, base=base, scale=1.0,
                        offset=offset)[0].transpose(1, 0, 2)


# ---------------------------------------------------------------- n-gram table (PLE)

class NGramTable:
    """mlx-serve-ngram: safetensors-framed 4-bit affine rows, group 32, bf16 scales/biases."""

    def __init__(self, path):
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
        self.bits = int(hdr["__metadata__"]["bits"])
        self.group = int(hdr["__metadata__"]["group_size"])
        base = 8 + n
        self.rows_n, self.wcols = hdr["weight"]["shape"]
        self.gcols = hdr["scales"]["shape"][1]
        self.off = {k: base + hdr[k]["data_offsets"][0] for k in ("weight", "scales", "biases")}
        self.wbytes, self.gbytes = self.wcols * 4, self.gcols * 2
        self.dim = self.wcols * 32 // self.bits
        self.fd = os.open(path, os.O_RDONLY)
        # Random 4 KiB-scale reads: issue them concurrently (pread releases the GIL) so the
        # drive's queue depth hides latency; a small row cache catches repeated n-grams.
        self.pool = ThreadPoolExecutor(32)
        self.cache = {}

    def _read(self, i):
        i = int(i)
        return (os.pread(self.fd, self.wbytes, self.off["weight"] + i * self.wbytes),
                os.pread(self.fd, self.gbytes, self.off["scales"] + i * self.gbytes),
                os.pread(self.fd, self.gbytes, self.off["biases"] + i * self.gbytes))

    def rows(self, ids):
        """ids (N,) int64 -> (N, dim) float32."""
        miss = [i for i in dict.fromkeys(ids.tolist()) if i not in self.cache]
        for i, r in zip(miss, self.pool.map(self._read, miss)):
            self.cache[i] = r
        if len(self.cache) > 1 << 20:
            self.cache.clear()
        raw = [self.cache[i] for i in ids.tolist()]
        w = np.frombuffer(b"".join(r[0] for r in raw), dtype=np.uint32).reshape(len(ids), self.wcols)
        s = (np.frombuffer(b"".join(r[1] for r in raw), dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        b = (np.frombuffer(b"".join(r[2] for r in raw), dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        s, b = s.reshape(len(ids), self.gcols), b.reshape(len(ids), self.gcols)
        per = 32 // self.bits
        q = (w[:, :, None] >> (np.arange(per, dtype=np.uint32) * self.bits)) & ((1 << self.bits) - 1)
        q = q.reshape(len(ids), -1, self.group).astype(np.float32)
        return (q * s[:, :, None] + b[:, :, None]).reshape(len(ids), -1)


def shift_ignore_eos(tokens, shift, eos):
    """Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos for one sequence."""
    if shift == 0:
        return tokens
    pos = np.arange(len(tokens))
    eos_pos = np.where(tokens == eos, pos, -1)
    prev = np.concatenate([[-1], np.maximum.accumulate(eos_pos)[:-1]])
    in_segment = pos - (prev + 1)
    src = pos - shift
    shifted = tokens[np.clip(src, 0, None)]
    return np.where((in_segment >= shift) & (src >= 0), shifted, eos)


# ---------------------------------------------------------------- model

class FlashNext:
    def __init__(self, path, ngram_path=None):
        self.cfg = cfg = Config.load(path)
        index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        shards = {}
        for k, f in index.items():
            if k.startswith(PREFIX) or k.startswith("language_model.lm_head"):
                shards.setdefault(f, []).append(k)
        raw = {}
        for f, keys in shards.items():
            data = mx.load(os.path.join(path, f))
            # `* 1` copies into MLX's own memory: loaded arrays are file-backed, and pages from the
            # (external) drive get evicted and re-read every token.
            part = {k: data[k] * 1 for k in keys}
            mx.eval(part)
            raw.update(part)
            del data
        self.w = raw
        self.ngram = NGramTable(ngram_path or os.path.join(path, "ngram_table.bin")) if cfg.ple_layers else None
        self._compile_blocks()
        # 68 GiB of weights leave ~10 GiB under the GPU working-set limit: keep MLX's buffer cache
        # small, or freed temporaries accumulate until macOS pages GPU memory and everything stalls.
        mx.set_cache_limit(2 << 30)
        # Wire the weights: decode reads all of them every token, so any page macOS swaps out under
        # memory pressure (browsers etc.) costs a disk read per token (measured: 28 -> ~1 tok/s).
        mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
        self.reset()

    def _compile_blocks(self):
        """mx.compile fuses each block's elementwise chain (the FP32 norms, gates and residual
        injections would otherwise be thousands of tiny kernels per token) and cuts graph building."""
        c = self.cfg

        def gr(h, nw, dq, uq, iw):
            L = h.shape[0]
            hn = rms_norm(h, nw, c.eps, group=c.hidden)
            mix = silu(linear(hn, dq) / c.hc)
            mix = mx.sigmoid(linear(mix, uq)).reshape(L, c.hc, c.hidden)
            mixed = (mix * hn.reshape(L, c.hc, c.hidden)).mean(axis=1)
            if iw is None:
                return mixed
            return mixed, 2 * mx.sigmoid(linear(hn, iw) / c.hc)

        def inject(h, o, inj):
            return h + (o[:, None, :] * inj[:, :, None]).reshape(h.shape[0], -1)

        kd = c.lin_k_heads * c.lin_k_dim

        def la_in(x, conv_state, qkvq, zq, bq, aq, convw):
            L = x.shape[0]
            qkv = linear(x, qkvq)
            conv_in = mx.concatenate([conv_state, qkv], axis=0)
            conv = silu(mx.conv1d(conv_in[None], convw.astype(mx.float32), groups=conv_in.shape[-1])[0])
            qh = conv[:, :kd].reshape(1, L, c.lin_k_heads, c.lin_k_dim)
            kh = conv[:, kd:2 * kd].reshape(1, L, c.lin_k_heads, c.lin_k_dim)
            vh = conv[:, 2 * kd:].reshape(1, L, c.lin_v_heads, c.lin_v_dim)
            qh, kh = normalize_qk(qh, kh, inv_scale=c.lin_k_dim ** -0.5, eps=1e-6)
            z = linear(x, zq).reshape(L, c.lin_v_heads, c.lin_v_dim)
            return qh, kh, vh, linear(x, aq)[None], linear(x, bq)[None], z, conv_in[-(c.conv_kernel - 1):]

        def la_out(out, z, nw, oq):
            L = z.shape[0]
            out = rms_norm(out[0], nw, c.eps) * mx.sigmoid(z)
            return linear(out.reshape(L, -1), oq)

        def route(x, gq):
            probs = mx.softmax(linear(x, gq), axis=-1, precise=True)
            idx = mx.argpartition(-probs, c.top_k - 1, axis=-1)[:, :c.top_k]
            sc = mx.take_along_axis(probs, idx, axis=-1)
            return idx, sc / sc.sum(axis=-1, keepdims=True)

        def shared(x, gq, uq, dq, sg):
            sh = linear(silu(linear(x, gq)) * linear(x, uq), dq)
            return mx.sigmoid(linear(x, sg)) * sh

        self._gr, self._inject = mx.compile(gr), mx.compile(inject)
        self._la_in, self._la_out = mx.compile(la_in), mx.compile(la_out)
        self._route, self._shared = mx.compile(route), mx.compile(shared)

    def q(self, name):
        """Quantized (w, scales, biases) or plain tensor by short name."""
        key = PREFIX + name if not name.startswith("language_model") else name
        if key + ".scales" in self.w:
            return (self.w[key + ".weight"], self.w[key + ".scales"], self.w[key + ".biases"])
        return self.w[key + ".weight"] if key + ".weight" in self.w else self.w[key]

    # -------- state

    def reset(self):
        c = self.cfg
        self.pos = 0
        self.state = []
        mx.clear_cache()
        for i, t in enumerate(c.layer_types):
            st = {}
            if t == "linear_attention":
                conv_dim = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim
                st["conv"] = mx.zeros((c.conv_kernel - 1, conv_dim))
                st["ssm"] = None
            else:
                # preallocated, head-major so attention reads the cache in place (grown in KV_STEP steps)
                st["k"] = mx.zeros((c.kv_heads, 0, c.head_dim))
                st["v"] = mx.zeros((c.kv_heads, 0, c.head_dim))
                st["idx_k"] = mx.zeros((0, c.idx_dim))       # raw indexer keys
                st["T"] = 0
                st["blk"] = mx.zeros((0, c.idx_dim))         # pooled, normed, roped block keys
            if i in c.ple_layers:
                st["ple_ctx"] = np.full(c.ngram_size - 1, c.eos, dtype=np.int64)
                st["ple_conv"] = mx.zeros(((c.ple_conv_kernel - 1) * c.ngram_size, c.hc * c.hidden))
            self.state.append(st)

    def _reserve(self, st, T):
        """Grow layer st's KV and indexer buffers to hold T tokens."""
        cap = st["k"].shape[1]
        if T <= cap:
            return
        new = ((T + KV_STEP - 1) // KV_STEP) * KV_STEP
        c = self.cfg
        def grow(a, axis):
            pad = list(a.shape)
            pad[axis] = new - cap
            return mx.concatenate([a, mx.zeros(pad)], axis=axis)
        st["k"], st["v"], st["idx_k"] = grow(st["k"], 1), grow(st["v"], 1), grow(st["idx_k"], 0)

    # -------- blocks

    def gated_residual(self, h, name, combine=True):
        return self._gr(h, self.q(name + ".hc_norm"), self.q(name + ".input_mix_weight_down"),
                        self.q(name + ".input_mix_weight_up"),
                        self.q(name + ".block_inject_weight") if combine else None)

    def linear_attn(self, x, i):
        st, p = self.state[i], f"layers.{i}.linear_attn."
        qh, kh, vh, a, b, z, st["conv"] = self._la_in(
            x, st["conv"], self.q(p + "in_proj_qkv"), self.q(p + "in_proj_z"), self.q(p + "in_proj_b"),
            self.q(p + "in_proj_a"), self.q(p + "conv1d"))
        out, st["ssm"] = gated_delta_update(qh, kh, vh, a, b, self.w[PREFIX + p + "A_log"],
                                            self.w[PREFIX + p + "dt_bias"], st["ssm"], None)
        # RMSNormGated (plain weight, sigmoid gate), then out_proj
        return self._la_out(out, z, self.q(p + "norm"), self.q(p + "out_proj"))

    def indexer_mask(self, x, i, start):
        """QSA token selection for queries at positions start..start+L-1; None when every visible
        token is selected (no more complete blocks than the budget allows)."""
        c, st, p = self.cfg, self.state[i], f"layers.{i}.self_attn.indexer."
        L = x.shape[0]
        qk = linear(x, self.q(p + "index_qk_proj"))
        q = qk[:, :c.idx_heads * c.idx_dim].reshape(L, c.idx_heads, c.idx_dim)
        raw_k = qk[:, c.idx_heads * c.idx_dim:]
        q = rope(rms_norm(q, self.q(p + "q_layernorm"), c.eps), start, c.rope_dims, c.rope_theta)
        T = start + L
        st["idx_k"][start:T] = raw_k
        idx_k = st["idx_k"][:T]
        nb = T // c.idx_ratio
        have = st["blk"].shape[0]
        if nb > have:  # pool newly completed blocks: mean of raw keys, k_layernorm, RoPE at block start
            grp = idx_k[have * c.idx_ratio:nb * c.idx_ratio].reshape(nb - have, c.idx_ratio, c.idx_dim)
            pooled = rms_norm(grp.mean(axis=1), self.q(p + "k_layernorm"), c.eps)
            roped = self._rope_rows(pooled, mx.arange(have, nb) * c.idx_ratio)
            st["blk"] = mx.concatenate([st["blk"], roped], axis=0)
        topk = c.idx_budget // c.idx_ratio
        if nb <= topk:
            return None
        scores = mx.maximum(mx.einsum("lhd,bd->lhb", q, st["blk"]), 0).sum(axis=1) / math.sqrt(c.idx_dim)
        if L == 1:
            # decode: the token indices themselves (top blocks' tokens, then the incomplete tail)
            top = mx.argpartition(-scores[0], topk - 1)[:topk]
            toks = (top[:, None] * c.idx_ratio + mx.arange(c.idx_ratio)[None]).reshape(-1)
            return mx.concatenate([toks, mx.arange(nb * c.idx_ratio, T)]) if T > nb * c.idx_ratio else toks
        qpos = mx.arange(start, start + L)
        nb_q = (qpos + 1) // c.idx_ratio  # complete blocks visible to each query
        valid = mx.arange(nb)[None, :] < nb_q[:, None]
        scores = mx.where(valid, scores, -mx.inf)
        part = mx.argpartition(scores, nb - topk, axis=-1)
        kth = mx.take_along_axis(scores, part[:, nb - topk:nb - topk + 1], axis=-1)
        blk_sel = (scores >= kth) & valid
        blk_sel = blk_sel | (valid & (nb_q[:, None] <= topk))
        tok_sel = mx.repeat(blk_sel, c.idx_ratio, axis=1)
        tok_sel = mx.concatenate([tok_sel, mx.zeros((L, T - tok_sel.shape[1]), dtype=mx.bool_)], axis=1)
        tpos = mx.arange(T)[None, :]
        tail = (tpos >= (nb_q * c.idx_ratio)[:, None]) & (tpos <= qpos[:, None])
        return tok_sel | tail

    def _rope_rows(self, rows, starts):
        # RoPE each pooled key at its own block-start position (positions are not contiguous)
        d, half = self.cfg.rope_dims, self.cfg.rope_dims // 2
        inv = self.cfg.rope_theta ** (-mx.arange(0, half, dtype=mx.float32) * 2 / d)
        ang = starts.astype(mx.float32)[:, None] * inv[None]
        cos, sin = mx.cos(ang), mx.sin(ang)
        x1, x2, rest = rows[:, :half], rows[:, half:d], rows[:, d:]
        return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], axis=1)

    def attention(self, x, i):
        c, st, p = self.cfg, self.state[i], f"layers.{i}.self_attn."
        L, start = x.shape[0], self.pos
        T = start + L
        self._reserve(st, T)
        sel = self.indexer_mask(x, i, start)
        qg = linear(x, self.q(p + "q_proj")).reshape(L, c.heads, 2 * c.head_dim)
        q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:].reshape(L, -1)
        k = linear(x, self.q(p + "k_proj")).reshape(L, c.kv_heads, c.head_dim)
        v = linear(x, self.q(p + "v_proj")).reshape(L, c.kv_heads, c.head_dim)
        q = rope(rms_norm(q, self.q(p + "q_norm"), c.eps), start, c.rope_dims, c.rope_theta)
        k = rope(rms_norm(k, self.q(p + "k_norm"), c.eps), start, c.rope_dims, c.rope_theta)
        st["k"][:, start:T] = k.transpose(1, 0, 2)
        st["v"][:, start:T] = v.transpose(1, 0, 2)
        qh = q.transpose(1, 0, 2)[None]
        if L == 1:  # decode: every cached token is causal; QSA picks a subset past the budget
            keys, vals = st["k"][None, :, :T], st["v"][None, :, :T]
            if sel is not None:
                keys, vals = mx.take(keys, sel, axis=2), mx.take(vals, sel, axis=2)
            out = mx.fast.scaled_dot_product_attention(qh, keys, vals, scale=c.head_dim ** -0.5)
        else:
            causal = mx.arange(T)[None, :] <= mx.arange(start, T)[:, None]
            mask = causal if sel is None else causal & sel
            out = mx.fast.scaled_dot_product_attention(qh, st["k"][None, :, :T], st["v"][None, :, :T],
                                                       scale=c.head_dim ** -0.5, mask=mask)
        out = out[0].transpose(1, 0, 2).reshape(L, -1)
        return linear(out * mx.sigmoid(gate), self.q(p + "o_proj"))

    def moe(self, x, i):
        p = f"layers.{i}.mlp."
        idx, sc = self._route(x, self.q(p + "gate"))
        y = self.experts(x, idx, sc, p)
        return y + self._shared(x, self.q(p + "shared_expert.gate_proj"), self.q(p + "shared_expert.up_proj"),
                                self.q(p + "shared_expert.down_proj"), self.q(p + "shared_expert_gate"))

    def experts(self, x, idx, sc, p):
        L = x.shape[0]
        gw, uw, dw = (self.q(p + f"switch_mlp.{n}") for n in ("gate_proj", "up_proj", "down_proj"))
        if L == 1:  # decode: fused FP32 kernel (stock gather_qmm is 4x slower with FP32 inputs)
            return fused_moe(x[0], idx[0].astype(mx.uint32), sc[0], gw, uw, dw)[None]
        # prompt chunks: (token, expert) pairs sorted by expert so each expert's weights are read
        # once per chunk (2.3x faster than unsorted; FP32 inputs then cost the same as bf16)
        K = idx.shape[1]
        flat = idx.flatten()
        order = mx.argsort(flat)
        xs = x[order // K][:, None, :]
        e = flat[order]
        g = mx.gather_qmm(xs, *gw, rhs_indices=e, transpose=True, group_size=64, bits=4, sorted_indices=True)
        u = mx.gather_qmm(xs, *uw, rhs_indices=e, transpose=True, group_size=64, bits=4, sorted_indices=True)
        y = mx.gather_qmm(silu(g) * u, *dw, rhs_indices=e, transpose=True, group_size=64, bits=4,
                          sorted_indices=True)[:, 0]
        y = y[mx.argsort(order)].reshape(L, K, -1)
        return (y * sc[..., None]).sum(axis=1)

    def ple(self, h, ids, i):
        c, st, p = self.cfg, self.state[i], f"layers.{i}.ple."
        L, H = len(ids), c.hidden
        e = self.w[PREFIX + p + "ple_embedding.layer_multipliers"].tolist()
        sizes = np.array(self.w[PREFIX + p + "ple_embedding.ngram_heads_vocab_sizes"].tolist(), dtype=np.int64)
        offs = np.array(self.w[PREFIX + p + "ple_embedding.ngram_heads_offsets"].tolist(), dtype=np.int64)
        hist = np.concatenate([st["ple_ctx"], ids])
        st["ple_ctx"] = hist[-(c.ngram_size - 1):]
        shifted = [shift_ignore_eos(hist, s, c.eos) for s in range(c.ngram_size)]
        blocks = []
        for n in range(2, c.ngram_size + 1):
            mixed = shifted[0] * e[0]
            for j in range(1, n):
                mixed = np.bitwise_xor(mixed, shifted[j] * e[j])
            sl = slice((n - 2) * c.heads_per_ngram, (n - 1) * c.heads_per_ngram)
            blocks.append(np.remainder(mixed[:, None], sizes[sl][None]) + offs[sl][None])
        rows = np.concatenate(blocks, axis=1)[-L:]  # (L, 16)
        emb = mx.array(self.ngram.rows(rows.reshape(-1)).reshape(L, -1))  # (L, ple_embed_dim)
        key = rms_norm(linear(emb, self.q(p + "key_proj")), self.q(p + "norm_key"), c.eps, group=H)
        val = linear(emb, self.q(p + "value_proj"))
        qry = rms_norm(h, self.q(p + "norm_query"), c.eps, group=H)
        g = (key.reshape(L, c.hc, H) * qry.reshape(L, c.hc, H)).sum(-1, keepdims=True) / math.sqrt(H)
        g = mx.sqrt(mx.maximum(mx.abs(g), 1e-6)) * mx.sign(g)
        gv = (mx.sigmoid(g) * val[:, None, :]).reshape(L, -1)
        gvn = rms_norm(gv, self.q(p + "norm_conv"), c.eps, group=H)
        # depthwise conv, kernel 4, dilation ngram_size, over [state, gvn]
        xp = mx.concatenate([st["ple_conv"], gvn], axis=0)
        st["ple_conv"] = xp[-st["ple_conv"].shape[0]:]
        w = self.q(p + "conv1d").astype(mx.float32)[:, :, 0]  # (C, K)
        d = c.ngram_size
        conv = sum(w[:, k][None] * xp[k * d:k * d + L] for k in range(c.ple_conv_kernel))
        return gv + silu(conv)

    # -------- forward

    def forward(self, ids, last_only=False):
        """ids: list/array of token ids continuing the cached sequence. Returns logits (L, vocab),
        or (1, vocab) for the last token only (prompt chunks need no other rows)."""
        c = self.cfg
        ids = np.asarray(ids, dtype=np.int64)
        L = len(ids)
        ew, es, eb = self.q("embed_tokens")
        tid = mx.array(ids)
        # FP32 scales: with the stored bf16 ones dequantize rounds the embedding to bf16
        emb = mx.dequantize(ew[tid], es[tid].astype(mx.float32), eb[tid].astype(mx.float32), group_size=64,
                            bits=qbits(ew, es))
        h = mx.tile(emb, (1, c.hc))
        for i, t in enumerate(c.layer_types):
            if i in c.ple_layers:
                h = h + self.ple(h, ids, i)
            x, inj = self.gated_residual(h, f"layers.{i}.attn_hyper_connection")
            o = self.linear_attn(x, i) if t == "linear_attention" else self.attention(x, i)
            h = self._inject(h, o, inj)
            x, inj = self.gated_residual(h, f"layers.{i}.mlp_hyper_connection")
            h = self._inject(h, self.moe(x, i), inj)
            if i % 8 == 7:
                mx.async_eval(h)  # let the GPU start while the rest of the graph is built
        if last_only:
            h = h[-1:]
        x = self.gated_residual(h, "hyper_connection_mixer", combine=False)
        self.pos += L
        return linear(x, self.q("language_model.lm_head"))
