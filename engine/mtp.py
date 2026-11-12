"""MTP speculative decoding for Qwen3.5/3.8 hybrid models on mlx-lm.

The model's native multi-token-prediction head (one full-attention layer that fuses the last hidden
state with the next token's embedding) drafts k tokens; the target verifies them in one forward pass,
which costs about the same memory traffic as one ordinary decode step. Sampling is exact: a
deterministic draft d is accepted with the target's probability p(d), and on rejection the token is
resampled from p with d removed, so the output distribution equals plain sampling (greedy: identical
tokens).

The linear-attention layers keep a recurrent state that cannot be trimmed, so during verification
they run the recurrence one token at a time and keep every intermediate state; a rejection restores
the state at the last accepted token. Attention KV caches are simply trimmed.
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5, qwen3_next
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

# ------------------------------------------------------------------ linear-attention state capture

_COMPILE = True
_CAPTURE = False  # set while verifying: GatedDeltaNet records the state after every token


def _capture_core(self, inputs, conv_state, state):
    """The layer's math for S tokens, also returning the conv and recurrent state after every token."""
    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = self.conv_kernel_size - 1
    conv_hist = tuple(conv_input[:, j:j + n_keep, :] for j in range(1, S + 1))  # conv state after j tokens
    conv_out = nn.silu(self.conv1d(conv_input))
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
        mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
        [self.num_k_heads, self.num_k_heads, self.num_v_heads],
        [self.head_k_dim, self.head_k_dim, self.head_v_dim])]
    q, k = normalize_qk(q, k, inv_scale=self.head_k_dim ** -0.5, eps=1e-6)
    outs, ssm_hist = [], []
    for j in range(S):
        o, state = gated_delta_update(q[:, j:j + 1], k[:, j:j + 1], v[:, j:j + 1], a[:, j:j + 1],
                                      b[:, j:j + 1], self.A_log, self.dt_bias, state, None)
        outs.append(o)
        ssm_hist.append(state)
    out = self.norm(mx.concatenate(outs, axis=1), z)
    return self.out_proj(out.reshape(B, S, -1)), conv_hist, tuple(ssm_hist)


def _gdn_call(self, inputs, mask=None, cache=None):
    if not (_CAPTURE and cache is not None):
        return _gdn_orig(self, inputs, mask, cache)
    B, S, _ = inputs.shape
    conv_state = cache[0] if cache[0] is not None else mx.zeros(
        (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
    state = cache[1] if cache[1] is not None else mx.zeros(
        (B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.float32)
    # compiled once per layer (weights are constants of the trace): verification always has the same
    # shapes, so this replaces ~100 us of graph building per layer per round and fuses the small ops
    if not _COMPILE:
        out, conv_hist, ssm_hist = _capture_core(self, inputs, conv_state, state)
    else:
        if "_capture" not in self.__dict__:
            self.__dict__["_capture"] = mx.compile(lambda x, c, st: _capture_core(self, x, c, st))
        out, conv_hist, ssm_hist = self.__dict__["_capture"](inputs, conv_state, state)
    cache[0], cache[1] = conv_hist[-1], ssm_hist[-1]
    cache.advance(S)
    cache.hist = (conv_hist, ssm_hist)
    return out


_gdn_orig = qwen3_5.GatedDeltaNet.__call__
qwen3_5.GatedDeltaNet.__call__ = _gdn_call


def rollback(cache, keep, total):
    """Keep the first `keep` of the `total` tokens just verified."""
    drop = total - keep
    for c in cache:
        if isinstance(c, KVCache):
            if drop:
                c.trim(drop)
        else:
            conv_hist, ssm_hist = c.hist
            c[0], c[1] = conv_hist[keep - 1], ssm_hist[keep - 1]
            c.hist = None


# ------------------------------------------------------------------ holes: queue a round before the last is known

def _attn_call(self, x, mask=None, cache=None):
    """Qwen3NextAttention.__call__, but when the cache carries holes: positions come from the GPU
    (cache.rope_pos) and attention skips the slots of rejected drafts (cache.hole_mask)."""
    hm = getattr(cache, "hole_mask", None)
    if hm is None:
        return _attn_orig(self, x, mask, cache)
    B, L, _ = x.shape
    queries, gate = mx.split(self.q_proj(x).reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    keys, values = self.k_proj(x), self.v_proj(x)
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    queries = self.rope(queries, offset=cache.rope_pos)
    keys = self.rope(keys, offset=cache.rope_pos)
    keys, values = cache.update_and_fetch(keys, values)
    out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=hm)
    return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate))


_attn_orig = qwen3_next.Qwen3NextAttention.__call__
qwen3_next.Qwen3NextAttention.__call__ = _attn_call


class Holes:
    """KV slots written since `base` by a set of attention caches that share positions. Every round
    writes a fixed number of slots; the ones holding rejected drafts stay in place but are masked out,
    and the true position of the next slot is kept on the GPU. So the next round can be built and
    queued before this round's acceptance is known. compact() restores the plain layout at the end."""

    def __init__(self, caches, base):
        self.caches, self.base = caches, base
        self.valid = mx.zeros((0,), mx.bool_)  # one per slot since base (lazy)
        self.pos = mx.array(base, mx.int32)   # true position of the next slot (lazy)
        self._prefix = {}

    def arm(self, L):
        """Before feeding L slots."""
        n = self.caches[0].offset - self.base
        if L not in self._prefix:
            self._prefix[L] = (mx.ones((L, self.base), mx.bool_), mx.tril(mx.ones((L, L), mx.bool_)))
        pre, tri = self._prefix[L]
        mask = mx.concatenate([pre, mx.broadcast_to(self.valid[None], (L, n)), tri], axis=1)
        for c in self.caches:
            c.rope_pos, c.hole_mask = self.pos, mask

    def commit(self, valid):
        """After feeding: which of the new slots hold real tokens (lazy bool vector)."""
        self.valid = mx.concatenate([self.valid, valid])
        self.pos = self.pos + valid.astype(mx.int32).sum()

    def save(self):
        return self.valid, self.pos, [c.offset for c in self.caches]

    def restore(self, snap):
        self.valid, self.pos, offs = snap
        for c, o in zip(self.caches, offs):
            c.offset = o

    def compact(self, keep):
        """keep: one bool per slot since base. Moves the kept slots together; drops the hole state."""
        for c in self.caches:
            c.rope_pos = c.hole_mask = None
        assert len(keep) == self.caches[0].offset - self.base
        idx = [self.base + i for i, k in enumerate(keep) if k]
        if len(idx) == len(keep):
            return
        ix = mx.array(idx, mx.int32)
        for c in self.caches:
            if idx:
                c.keys[..., self.base:self.base + len(idx), :] = c.keys[..., ix, :]
                c.values[..., self.base:self.base + len(idx), :] = c.values[..., ix, :]
            c.offset = self.base + len(idx)


# ------------------------------------------------------------------ the MTP head

class MTP(nn.Module):
    def __init__(self, args):
        super().__init__()
        h, eps = args.hidden_size, args.rms_norm_eps
        self.pre_fc_norm_embedding = nn.RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(h, eps=eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.layers = [qwen3_5.DecoderLayer(args, args.full_attention_interval - 1)]  # a full-attention layer
        self.norm = nn.RMSNorm(h, eps=eps)

    def __call__(self, emb, hidden, cache):
        x = self.fc(mx.concatenate([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hidden)], -1))
        x = self.layers[0](x, create_attention_mask(x, cache), cache)
        return self.norm(x)


def find_mtp(path):
    f = os.path.join(path, "mtp", "model.safetensors")
    return f if os.path.exists(f) else None


def load_mtp(path, text_model):
    weights = mx.load(find_mtp(path))
    cfg = json.load(open(os.path.join(path, "config.json")))
    q = cfg.get("quantization") or {}
    m = MTP(text_model.args)
    if q:
        nn.quantize(m, group_size=q["group_size"], bits=q["bits"],
                    class_predicate=lambda p, _: f"{p}.scales" in weights)
    m.load_weights(list(weights.items()))
    # run in the target's activation dtype
    dt = text_model.model.embed_tokens.scales.dtype if hasattr(text_model.model.embed_tokens, "scales") \
        else text_model.model.embed_tokens.weight.dtype
    m.set_dtype(dt)
    mx.eval(m.parameters())
    return m


# ------------------------------------------------------------------ sampling

def filtered_probs(logits, temp, top_p, top_k):
    """(n, V) logits -> (n, V) float32 probabilities after temperature, top-k and top-p.
    With top-k, top-p is applied within those k candidates only (no sort over the 248K vocabulary)."""
    lg = logits.astype(mx.float32) / temp
    if top_k and top_k < lg.shape[-1]:
        idx = mx.argpartition(-lg, top_k - 1, axis=-1)[:, :top_k]
        p = mx.softmax(mx.take_along_axis(lg, idx, axis=-1), axis=-1)
        if top_p and top_p < 1.0:
            order = mx.argsort(-p, axis=-1)
            sp = mx.take_along_axis(p, order, axis=-1)
            sp = mx.where((mx.cumsum(sp, axis=-1) - sp) >= top_p, 0, sp)
            idx, p = mx.take_along_axis(idx, order, axis=-1), sp / sp.sum(-1, keepdims=True)
        return mx.put_along_axis(mx.zeros(lg.shape, mx.float32), idx, p, axis=-1)
    p = mx.softmax(lg, axis=-1)
    if top_p and top_p < 1.0:
        order = mx.argsort(-p, axis=-1)
        sp = mx.take_along_axis(p, order, axis=-1)
        cut = (mx.cumsum(sp, axis=-1) - sp) >= top_p  # drop tokens once the mass before them reaches top_p
        sp = mx.where(cut, 0, sp)
        p = mx.zeros_like(p)
        p = mx.put_along_axis(p, order, sp, axis=-1)
        p = p / p.sum(-1, keepdims=True)
    return p


def categorical(p):
    """(V,) probabilities -> int32 token."""
    return mx.random.categorical(mx.log(p + 1e-30)).astype(mx.int32)


def accept_sampled(p0, p1, d, q):
    """A draft d sampled from q, checked against the target: p0 is the target's distribution at d's
    position, p1 the one after d. Returns (accepted, next token). Exact for any q: accept with
    probability min(1, p0[d] / q[d]); on rejection resample from max(p0 - q, 0), renormalized. Sampling
    the draft (instead of taking its argmax) accepts far more often when the target samples too."""
    acc = mx.random.uniform() * q[d] < p0[d]
    return acc, mx.where(acc, categorical(p1), categorical(mx.maximum(p0 - q, 0)))


class Speculator:
    """Drafts with the MTP head and verifies with the target. One instance per conversation state."""

    def __init__(self, model, mtp, k=3):
        self.lm = model.language_model if hasattr(model, "language_model") else model
        self.mtp, self.k = mtp, k
        self.mtp_cache = KVCache()
        self.accepted = self.drafted = 0

    def target(self, ids, cache):
        h = self.lm.model(ids, cache)
        return self.lm.lm_head(h), h

    def mtp_step(self, hidden, tokens):
        """hidden: (1, n, H) target/MTP hidden states; tokens: (n,) next tokens -> logits, mtp hidden."""
        m = self.mtp(self.lm.model.embed_tokens(tokens[None]), hidden, self.mtp_cache)
        return self.lm.lm_head(m), m

    def prefill_mtp(self, hidden, next_tokens):
        """Feed (h_i, t_{i+1}) for the prompt so the MTP layer has the context; returns first draft."""
        lg, m = self.mtp_step(hidden, next_tokens)
        return lg[0, -1], m[:, -1:]

    def generate(self, cache, hidden, logits, y, max_tokens, temp=0.0, top_p=1.0, top_k=0):
        """cache: target cache after the prompt; hidden/logits: target outputs at the last prompt
        position; y: the sampled next token (int). Yields tokens. MTP must already be prefilled
        (prefill_mtp) with pairs ending in (last hidden, y)."""
        greedy = not temp
        draft_lg, m = self.mtp_last
        n = 0
        while n < max_tokens:
            # draft k tokens (greedy from the MTP head; chained on its own hidden state)
            drafts = [mx.argmax(draft_lg)]
            for _ in range(self.k - 1):
                lg, m = self.mtp_step(m, drafts[-1][None])
                drafts.append(mx.argmax(lg[0, -1]))
            seq = mx.concatenate([mx.array([y]), mx.stack(drafts)])
            global _CAPTURE
            _CAPTURE = True
            lg, h = self.target(seq[None], cache)
            _CAPTURE = False
            lg = lg[0]
            if greedy:
                pick = mx.argmax(lg, axis=-1)
                mx.eval(pick, seq)
                pick, d = pick.tolist(), seq.tolist()[1:]
                a = 0
                while a < self.k and d[a] == pick[a]:
                    a += 1
                nxt = pick[a]
            else:
                p = filtered_probs(lg, temp, top_p, top_k)
                u = mx.random.uniform(shape=(self.k,))
                d = seq[1:]
                pd = mx.take_along_axis(p[:-1], d[:, None], axis=-1)[:, 0]
                ok = (u < pd).tolist()
                d = d.tolist()
                a = 0
                while a < self.k and ok[a]:
                    a += 1
                if a < self.k:  # resample from p without the rejected draft
                    row = p[a] * (mx.arange(p.shape[-1]) != d[a])
                    nxt = mx.random.categorical(mx.log(row + 1e-30)).item()
                else:
                    nxt = mx.random.categorical(mx.log(p[a] + 1e-30)).item()
            self.drafted += self.k
            self.accepted += a
            keep = a + 1
            rollback(cache, keep, self.k + 1)
            # MTP cache: drop the chained draft entries, then add the verified pairs
            self.mtp_cache.trim(self.k - 1)
            out = [y] + d[:a]
            for t in out:
                yield t
                n += 1
                if n >= max_tokens:
                    return
            y = nxt
            lgm, m = self.mtp_step(h[:, :keep], mx.array(d[:a] + [y]))
            draft_lg, m = lgm[0, -1], m[:, -1:]

    def start(self, prompt_hidden, prompt_ids, y):
        """Prefill the MTP layer from the prompt's hidden states (1, N, H), ids (N,), and next token y."""
        nxt = mx.concatenate([prompt_ids[1:], mx.array([y])])
        self.mtp_last = self.prefill_mtp(prompt_hidden, nxt)
