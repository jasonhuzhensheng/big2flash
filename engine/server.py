"""Anthropic Messages API server for the big2flash engine, so Claude Code can use the local model.

  .venv/bin/python engine/server.py [--port 8080]

  ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=local claude

Endpoints: POST /v1/messages (streaming and not, tools, thinking), POST /v1/messages/count_tokens,
GET /v1/models. One request runs at a time (single thread); others wait their turn.

Prompt cache: Claude Code resends the whole conversation (~20K tokens of system prompt and tool
definitions before the first message) on every call. The engine's state is recurrent (linear
attention), so it can't be cut back to an arbitrary length; instead each session keeps snapshots
of the small per-layer state at a few positions (every 4,096 prompt tokens and at the end of each
prompt). The attention KV buffers are append-only, so rolling back to a snapshot only rewinds the
position. A request continues the session and snapshot with the longest matching token prefix.
"""
import argparse, json, os, re, sys, threading, time, uuid
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # no "PyTorch was not found" notice: only the tokenizer is used
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MODEL = "/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw"
SNAP_EVERY = 4096
ATTN_BUDGET = 2 << 30  # bytes of attention scores per prefill step (MLX materializes them at head_dim 256)
HEADROOM = 6 << 30     # GPU memory kept free beyond weights + KV (activations, snapshots, buffer cache)


def kv_bytes_per_token(t):
    """KV cache bytes one token adds across the full-attention layers (bf16 K and V, plus FlashNext's
    fp32 indexer keys); the linear-attention layers keep fixed-size state."""
    full = sum(x == "full_attention" for x in t.get("layer_types") or []) \
        or t["num_hidden_layers"] // t.get("full_attention_interval", 1)
    return full * (2 * t["num_key_value_heads"] * t["head_dim"] * 2 + 4 * t.get("indexer_head_dim", 0))
BUFFERS = ("k", "v", "idx_k")  # append-only attention buffers, shared by all snapshots of a session


# ============================================================ request -> prompt

def text_of(content):
    """Text of a tool_result / document content: a string or a list of blocks."""
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if b.get("type") == "text":
            out.append(b["text"])
        elif b.get("type") == "image":
            out.append("[image omitted: this model reads text only]")
    return "\n".join(out)


def to_chat(req):
    """Anthropic request -> (messages, tools) in the shape the Qwen chat template expects."""
    msgs = []
    system = req.get("system")
    if system:
        msgs.append({"role": "system", "content": system if isinstance(system, str)
                     else "\n\n".join(b["text"] for b in system if b.get("type") == "text")})
    for m in req["messages"]:
        content = m["content"]
        if m["role"] == "system":  # Claude Code sends mid-conversation system messages; the template
            # allows a system message only first, so give it its own user turn (earlier turns, and so
            # the prefix cache, stay unchanged)
            msgs.append({"role": "user", "content": f"<system>\n{text_of(content)}\n</system>"})
            continue
        if isinstance(content, str):
            msgs.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text, reasoning, calls = [], [], []
            for b in content:
                t = b.get("type")
                if t == "text":
                    text.append(b["text"])
                elif t == "thinking":
                    reasoning.append(b.get("thinking", ""))
                elif t == "tool_use":
                    calls.append({"type": "function", "function": {"name": b["name"], "arguments": b.get("input") or {}}})
            msg = {"role": "assistant", "content": "\n\n".join(text)}
            if reasoning:
                msg["reasoning_content"] = "\n\n".join(reasoning)
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
            continue
        text = []
        for b in content:
            t = b.get("type")
            if t == "tool_result":
                r = text_of(b.get("content"))
                msgs.append({"role": "tool", "content": ("Error: " + r) if b.get("is_error") else r})
            elif t == "text":
                text.append(b["text"])
            elif t == "image":
                text.append("[image omitted: this model reads text only]")
            elif t == "document":
                src = b.get("source", {})
                text.append(src.get("data", "") if src.get("type") == "text" else "[document omitted]")
        if text:
            msgs.append({"role": "user", "content": "\n\n".join(text)})
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t.get("input_schema", {})}}
             for t in req.get("tools") or [] if "input_schema" in t]
    return msgs, tools


def thinking_mode(req, policy):
    """(enable_thinking, reasoning_effort) for the template."""
    th = req.get("thinking") or {}
    on = policy == "on" or (policy == "auto" and th.get("type") in ("enabled", "adaptive"))
    if not on:
        return False, None
    budget = th.get("budget_tokens") or 0
    return True, "low" if budget and budget < 8000 else "medium" if budget and budget < 24000 else "xhigh"


def forced_prefix(req):
    """tool_choice any/tool: start the reply inside a tool call."""
    tc = req.get("tool_choice") or {}
    if not req.get("tools"):
        return ""
    if tc.get("type") == "tool":
        return f"<tool_call>\n<function={tc['name']}>\n"
    if tc.get("type") == "any":
        return "<tool_call>\n<function="
    return ""


# ============================================================ model output -> content blocks

def parse_call(block, schemas):
    """'<function=NAME>...<parameter=P>\\nV\\n</parameter>...</function>' -> (name, input dict)."""
    m = re.search(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", block, re.S)
    if not m:
        return None
    name = m.group(1).strip()
    props = (schemas.get(name) or {}).get("properties", {})
    args = {}
    for p in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", m.group(2), re.S):
        key, raw = p.group(1).strip(), p.group(2)
        typ = (props.get(key) or {}).get("type")
        if typ == "string" or (isinstance(typ, list) and "string" in typ and not raw.lstrip()[:1] in "[{"):
            args[key] = raw
            continue
        try:
            args[key] = json.loads(raw)
        except ValueError:
            args[key] = raw
    return name, args


class OutputParser:
    """Turns the growing decoded text into Anthropic content-block events:
    ("thinking", text) / ("text", text) deltas and ("tool", name, input) for each finished call.
    Text that might be the start of a tag (or trailing whitespace) is held back until it's settled."""

    def __init__(self, schemas, thinking, prefix=""):
        self.schemas = schemas
        self.mode = "thinking" if thinking else "text"
        self.buf = prefix  # text not yet turned into events
        self.fresh = True  # at the start of a block: drop leading whitespace

    def _hold(self, s, tag):
        """Length of s that is safe to emit: not a possible prefix of tag, not trailing whitespace."""
        n = len(s)
        for k in range(min(len(tag) - 1, n), 0, -1):
            if tag.startswith(s[n - k:]):
                n -= k
                break
        return len(s[:n].rstrip())

    def feed(self, new, final=False):
        self.buf += new
        events = []
        while True:
            if self.mode == "thinking":
                i = self.buf.find("</think>")
                if i < 0:
                    n = len(self.buf.rstrip()) if final else self._hold(self.buf, "</think>")
                    self._emit(events, "thinking", self.buf[:n])
                    self.buf = self.buf[n:]
                    return events
                self._emit(events, "thinking", self.buf[:i].rstrip())
                self.buf, self.mode, self.fresh = self.buf[i + len("</think>"):], "text", True
            elif self.mode == "text":
                i = self.buf.find("<tool_call>")
                if i < 0:
                    n = len(self.buf.rstrip()) if final else self._hold(self.buf, "<tool_call>")
                    self._emit(events, "text", self.buf[:n])
                    self.buf = self.buf[n:]
                    return events
                self._emit(events, "text", self.buf[:i].rstrip())
                self.buf, self.mode = self.buf[i + len("<tool_call>"):], "tool"
            else:
                i = self.buf.find("</tool_call>")
                if i < 0 and not final:
                    return events
                call = parse_call(self.buf if i < 0 else self.buf[:i], self.schemas)
                if call:
                    events.append(("tool",) + call)
                self.buf = "" if i < 0 else self.buf[i + len("</tool_call>"):]
                self.mode, self.fresh = "text", True
                if i < 0:
                    return events

    def _emit(self, events, kind, s):
        if self.fresh:
            s = s.lstrip()
            if not s:  # still only whitespace: stay at the block start
                return
        self.fresh = False
        if s:
            events.append((kind, s))


# ============================================================ engine sessions

class Session:
    def __init__(self, state, pos=0):
        self.state, self.pos = state, pos
        self.tokens = []      # tokens fed to the model so far (len == pos)
        self.snaps = []       # (pos, per-layer small state)
        self.used = time.time()


class FlashBackend:
    """FlashNext: state is a list of per-layer dicts."""
    chunk = 512  # prefill tokens per forward pass

    def __init__(self, path):
        from flash_next import FlashNext
        self.m = FlashNext(path)

    def __getattr__(self, name):
        return getattr(self.m, name)

    def __setattr__(self, name, value):
        if name in ("state", "pos"):
            setattr(self.m, name, value)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def snapshot(state):
        """The per-layer state without the shared append-only buffers. MLX arrays are immutable and
        the engine replaces (rather than edits) every one of these, so references suffice."""
        return [{k: v for k, v in st.items() if k not in BUFFERS} for st in state]

    @staticmethod
    def restore(state, snap):
        for st, small in zip(state, snap):
            st.update(small)

    @staticmethod
    def arrays(state):
        return [v for st in state for v in st.values() if isinstance(v, mx.array)]


class MlxLmBackend:
    """Any model mlx-lm can load; state is its prompt cache (KVCache / ArraysCache per layer)."""
    chunk = 2048  # mlx-lm's own prefill step: measured faster than 512 on dense models

    def __init__(self, path, mtp_k=1):
        from mlx_lm import load
        from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache
        import mtp
        self.model, _ = load(path)
        self.lm = getattr(self.model, "language_model", self.model)
        self.n = len(make_prompt_cache(self.model))
        # MTP speculative decoding when the model ships its multi-token-prediction head (mtp/ sidecar).
        # The session state then carries two extra entries: the head's KV cache, and the last target
        # hidden state still waiting for its next token (so prefill can feed the head as it goes).
        self.head = mtp.load_mtp(path, self.lm) if mtp_k and mtp.find_mtp(path) else None
        self.k = mtp_k
        extra = (lambda: [KVCache(), ArraysCache(1)]) if self.head else (lambda: [])
        self._make = lambda: make_prompt_cache(self.model) + extra()
        self.reset()

    def reset(self):
        self.state, self.pos = self._make(), 0
        mx.clear_cache()

    def forward(self, ids, last_only=False):
        ids = mx.array(ids)
        if not self.head:
            out = self.model(ids[None], cache=self.state)[0]
            self.pos += len(ids)
            return out[-1:] if last_only else out
        h = self.lm.model(ids[None], cache=self.state[:self.n])
        self._feed_head(h, ids)
        self.pos += len(ids)
        return self.lm.lm_head(h[:, -1:] if last_only else h)[0]

    def _feed_head(self, h, ids):
        """Give the MTP head the pairs (hidden_i, token_i+1) these tokens complete; keep the last
        hidden state pending until its next token is known. No output head: only its cache matters."""
        head_cache, pending = self.state[self.n], self.state[self.n + 1]
        if pending[0] is not None:
            hin, toks = mx.concatenate([pending[0], h[:, :-1]], axis=1), ids
        else:
            hin, toks = h[:, :-1], ids[1:]
        if toks.size:
            self.head(self.lm.model.embed_tokens(toks[None]), hin, head_cache)
        pending[0] = h[:, -1:]

    def generate_spec(self, s, logits, max_tokens, sampling, eos):
        """Speculative decoding (see mtp.py). Yields tokens; keeps s.tokens/s.pos (and the head's
        cache) exactly in step with what the target has processed, eos included, like Runner.generate."""
        import mtp
        temp, top_p, top_k = sampling["temperature"] or 0.0, sampling["top_p"] or 1.0, sampling["top_k"] or 0
        cache, head_cache, pending = self.state[:self.n], self.state[self.n], self.state[self.n + 1]
        emb, lm_head, k = self.lm.model.embed_tokens, self.lm.lm_head, self.k

        def sample(lg):  # (V,) -> token
            if not temp:
                return mx.argmax(lg).item()
            return mx.random.categorical(mx.log(mtp.filtered_probs(lg[None], temp, top_p, top_k)[0] + 1e-30)).item()

        y = sample(logits)
        m = self.head(emb(mx.array([[y]])), pending[0], head_cache)  # pair (last hidden, y)
        n = 0
        while True:
            drafts = [mx.argmax(lm_head(m)[0, -1])]
            for _ in range(k - 1):
                m = self.head(emb(drafts[-1].reshape(1, 1)), m, head_cache)
                drafts.append(mx.argmax(lm_head(m)[0, -1]))
            seq = mx.concatenate([mx.array([y]), mx.stack(drafts)])
            mtp._CAPTURE = True
            h = self.lm.model(seq[None], cache=cache)
            mtp._CAPTURE = False
            lg = lm_head(h)[0]
            if not temp:
                pick = mx.argmax(lg, axis=-1)
                mx.eval(pick, seq)  # one sync per round: drafts and verification together
                pick, d = pick.tolist(), seq.tolist()[1:]
                a = 0
                while a < k and d[a] == pick[a]:
                    a += 1
                nxt = pick[a]
            else:
                p = mtp.filtered_probs(lg, temp, top_p, top_k)
                ok = mx.random.uniform(shape=(k,)) < p[mx.arange(k), seq[1:]]
                mx.eval(ok, seq)
                ok, d = ok.tolist(), seq.tolist()[1:]
                a = 0
                while a < k and ok[a]:
                    a += 1
                row = p[a] * (mx.arange(p.shape[-1]) != d[a]) if a < k else p[a]
                nxt = mx.random.categorical(mx.log(row + 1e-30)).item()
            out = [y] + d[:a]
            # how many of these to keep: stop at eos (kept in the state, not yielded) or max_tokens
            e, stop = len(out), False
            for i, t in enumerate(out):
                if t in eos or n + i + 1 >= max_tokens:
                    e, stop = i + 1, True
                    break
            mtp.rollback(cache, e, k + 1)
            head_cache.trim(k - 1)  # the chained draft entries; the (last hidden, y) pair stays
            s.tokens += out[:e]
            s.pos += e
            self.pos = s.pos
            for t in out[:e]:
                if t in eos:
                    break
                yield t
                n += 1
            if stop:
                if e > 1:  # complete the pairs inside the kept tokens; the last hidden waits
                    self.head(emb(mx.array([out[1:e]])), h[:, :e - 1], head_cache)
                pending[0] = h[:, e - 1:e]
                return
            y = nxt
            m = self.head(emb(mx.array([out[1:] + [y]])), h[:, :e], head_cache)[:, -1:]

    def generate_pipe(self, s, logits, max_tokens, sampling, eos):
        """generate_spec for k=1, pipelined: round r+1 (verify [y, draft], feed the head, next draft)
        is built from round r's outputs while they are still lazy and queued before round r's tokens
        are read, so the GPU never waits on Python. Acceptance, the next token, positions and the
        linear-attention state choice all stay on the GPU; rejected drafts become masked KV slots
        (mtp.Holes) that compact() removes when generation ends. Same tokens as generate_spec."""
        import mtp
        temp, top_p, top_k = sampling["temperature"] or 0.0, sampling["top_p"] or 1.0, sampling["top_k"] or 0
        cache, head_cache, pending = self.state[:self.n], self.state[self.n], self.state[self.n + 1]
        emb, lm_head = self.lm.model.embed_tokens, self.lm.lm_head
        fa = [c for c in cache if hasattr(c, "offset")]
        gdn = [c for c in cache if not hasattr(c, "offset")]
        i32 = mx.int32

        def probs(lg):  # (n, V) logits -> sampling distribution
            return mtp.filtered_probs(lg, temp, top_p, top_k)

        def draft(hidden):  # head output (1, H) -> (draft token, its distribution or None)
            lg = lm_head(hidden)
            if not temp:
                return mx.argmax(lg[0]).astype(i32), None
            q = probs(lg)[0]
            return mtp.categorical(q), q

        y = mx.argmax(logits).astype(i32) if not temp else mtp.categorical(probs(logits[None])[0])
        m = self.head(emb(y.reshape(1, 1)), pending[0], head_cache)  # pair (last hidden, y)
        d, q = draft(m[:, -1])
        th, hh = mtp.Holes(fa, fa[0].offset), mtp.Holes([head_cache], head_cache.offset)
        one = mx.array([True])

        def build(y, d, q):
            th.arm(2)
            mtp._CAPTURE = True
            h = self.lm.model(mx.stack([y, d])[None], cache=cache)
            mtp._CAPTURE = False
            lg = lm_head(h)[0]
            if not temp:
                pick = mx.argmax(lg, axis=-1).astype(i32)
                acc = pick[0] == d
                nxt = mx.where(acc, pick[1], pick[0])
            else:  # d was sampled from the head's q: exact acceptance (mtp.accept_sampled)
                p = probs(lg)
                acc, nxt = mtp.accept_sampled(p[0], p[1], d, q)
            hist = []
            for c in gdn:
                (ch, sh), c.hist = c.hist, None
                hist.append((ch, sh))
                c[0], c[1] = mx.where(acc, ch[1], ch[0]), mx.where(acc, sh[1], sh[0])
            valid = mx.concatenate([one, acc[None]])
            th.commit(valid)
            # head pairs: (h_y, token after y) and (h_d, nxt), the second real only if d was accepted
            hh.arm(2)
            m = self.head(emb(mx.stack([mx.where(acc, d, nxt), nxt])[None]), h, head_cache)
            hh.commit(valid)
            d2, q2 = draft(mx.where(acc, m[:, 1], m[:, 0]))
            r = {"y": y, "d": d, "acc": acc, "nxt": nxt, "d2": d2, "q2": q2, "h": h, "hist": hist}
            mx.async_eval(y, d, acc, nxt, d2)
            return r

        def save():
            return th.save(), hh.save(), [(c[0], c[1]) for c in gdn]

        def restore(snap):
            t, hd, g = snap
            th.restore(t)
            hh.restore(hd)
            for c, (a, b) in zip(gdn, g):
                c[0], c[1] = a, b

        keep_t, keep_h = [], []
        cur, done, snap = build(y, d, q), False, None
        try:
            n = 0
            while True:
                snap = save()
                nxt_round = build(cur["nxt"], cur["d2"], cur["q2"])
                yv, dv, av = cur["y"].item(), cur["d"].item(), cur["acc"].item()
                out = [yv, dv] if av else [yv]
                e = len(out)
                for i, t in enumerate(out):
                    if t in eos or n + i + 1 >= max_tokens:
                        e, done = i + 1, True
                        break
                s.tokens += out[:e]
                s.pos += e
                self.pos = s.pos
                if done:
                    break
                for t in out:
                    yield t
                    n += 1
                keep_t += [True, bool(av)]
                keep_h += [True, bool(av)]
                cur = nxt_round
            for t in out[:e]:
                if t in eos:
                    break
                yield t
        finally:
            # undo the round queued past the end, then finish round `cur` with its first e tokens kept
            # (also when the consumer stops early: s.tokens then already holds all of cur's tokens)
            if snap is None:  # failed before the first round was read
                return
            restore(snap)
            if not done:
                e = len(out)
            keep_t += [True, e == 2]
            keep_h += [e == 2, False]  # (h_y, d) only if d is kept; (h_d, nxt) never: nxt isn't
            if e == 1 and av:  # d was accepted but is cut: state after y only
                for c, (ch, sh) in zip(gdn, cur["hist"]):
                    c[0], c[1] = ch[0], sh[0]
            th.compact(keep_t)
            hh.compact(keep_h)
            pending[0] = cur["h"][:, e - 1:e]

    @staticmethod
    def snapshot(state):
        # KV caches only append (rolling back = lowering the offset); recurrent ArraysCache entries
        # are replaced on every update, so keeping references is enough
        return [("kv", c.offset) if hasattr(c, "offset") else ("arr", list(c.cache)) for c in state]

    @staticmethod
    def restore(state, snap):
        for c, (kind, v) in zip(state, snap):
            if kind == "kv":
                c.offset = v
            else:
                c.cache = list(v)

    @staticmethod
    def arrays(state):
        out = []
        for c in state:
            vals = [c.keys, c.values] if hasattr(c, "offset") else c.cache
            out += [v for v in vals if isinstance(v, mx.array)]
        return out


class Runner:
    def __init__(self, path, sessions, max_context, snaps, wired_margin_gb=2.0, cache_gb=0.5, mtp_k=1):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        t0 = time.perf_counter()
        cfg = json.load(open(os.path.join(path, "config.json")))
        kind = cfg.get("text_config", cfg).get("model_type", cfg.get("model_type"))
        self.m = FlashBackend(path) if kind == "qwen4_exp" else MlxLmBackend(path, mtp_k)
        spec = f", MTP speculative decoding k={self.m.k}" if getattr(self.m, "head", None) is not None else ""
        print(f"{kind} loaded in {time.perf_counter() - t0:.0f} s ({type(self.m).__name__}{spec})", flush=True)
        weights = mx.get_active_memory()
        mx.set_cache_limit(int(cache_gb * 2**30))
        gc = json.load(open(os.path.join(path, "generation_config.json")))
        self.defaults = {k: gc.get(k) for k in ("temperature", "top_p", "top_k")}
        self.eos = set(gc["eos_token_id"] if isinstance(gc["eos_token_id"], list) else [gc["eos_token_id"]])
        t = cfg.get("text_config", cfg)
        self.q_heads = t["num_attention_heads"]
        fits = (mx.device_info()["max_recommended_working_set_size"] - weights - HEADROOM) // kv_bytes_per_token(t)
        if max_context > fits:
            max_context = max(4096, fits // 1024 * 1024)
            print(f"max context lowered to fit GPU memory", flush=True)
        print(f"context limit {max_context} tokens", flush=True)
        # Wire the weights plus the KV cache the context limit allows: GPU memory outside the wired set
        # reads at ~40 GB/s instead of ~350. Wiring all the way to the GPU limit instead starves the rest
        # of the system once a long agent context grows (every new allocation waits on page eviction).
        mx.set_wired_limit(min(mx.device_info()["max_recommended_working_set_size"],
                               weights + max_context * kv_bytes_per_token(t) + int(wired_margin_gb * 2**30)))
        self.max_sessions, self.max_context, self.max_snaps = sessions, max_context, snaps
        self.im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        self.sessions = []
        self.lock = threading.Lock()

    # -------- prompt
    def render(self, req, policy):
        msgs, tools = to_chat(req)
        prefix = forced_prefix(req)
        think, effort = thinking_mode(req, policy) if not prefix else (False, None)
        kw = {"enable_thinking": think}
        if effort:
            kw["reasoning_effort"] = effort
        text = self.tok.apply_chat_template(msgs, tools=tools or None, tokenize=False,
                                            add_generation_prompt=True, **kw)
        return self.tok(text + prefix, add_special_tokens=False)["input_ids"], think, prefix

    # -------- sessions
    def _new_state(self):
        self.m.reset()
        return self.m.state

    def _pick(self, ids):
        """(session, usable prefix length): the longest cached prefix of ids we can continue from."""
        best, best_n = None, 0
        for s in self.sessions:
            n = 0
            lim = min(len(s.tokens), len(ids) - 1)  # at least one token must be fed to get logits
            a, b = np.asarray(s.tokens[:lim]), np.asarray(ids[:lim])
            lcp = int(np.argmin(a == b)) if lim and not (a == b).all() else lim
            if lcp == len(s.tokens):
                n = lcp
            else:
                n = max([p for p, _ in s.snaps if p <= lcp], default=0)
            if n > best_n:
                best, best_n = s, n
        if best is None:
            if len(self.sessions) < self.max_sessions:
                best = Session(self._new_state())
                self.sessions.append(best)
            else:
                best = min(self.sessions, key=lambda s: s.used)
                best.state, best.pos, best.tokens, best.snaps = self._new_state(), 0, [], []
        self._make_room(best)
        if best_n and best_n < best.pos:  # roll back to the snapshot
            snap = [st for p, st in best.snaps if p == best_n][0]
            self.m.restore(best.state, snap)
            best.pos, best.tokens = best_n, best.tokens[:best_n]
            best.snaps = [(p, st) for p, st in best.snaps if p <= best_n]
        best.used = time.time()
        return best, best_n

    def _make_room(self, keep):
        """Drop other sessions' caches (least recently used first) while all of them together
        would hold more than max_context tokens of KV."""
        for s in sorted(self.sessions, key=lambda s: s.used):
            if sum(x.pos for x in self.sessions) <= self.max_context:
                return
            if s is not keep and s.pos:
                s.state, s.pos, s.tokens, s.snaps = self._new_state(), 0, [], []

    def _snapshot(self, s):
        if s.snaps and s.snaps[-1][0] == s.pos:
            return
        s.snaps.append((s.pos, self.m.snapshot(s.state)))
        while len(s.snaps) > self.max_snaps:  # thin out: drop the one closest to its predecessor
            gaps = [s.snaps[i][0] - s.snaps[i - 1][0] for i in range(1, len(s.snaps) - 1)]
            del s.snaps[1 + int(np.argmin(gaps))]

    # -------- run
    def prefill(self, s, ids, start, tick):
        """Feed ids[start:] to session s; returns the last logits. tick() is called between chunks."""
        m = self.m
        m.state, m.pos = s.state, s.pos
        logits = None
        pos = start
        # The next request re-renders this turn's reply, which need not match the generated tokens,
        # so also snapshot where the assistant turn starts: everything before it is reused next turn.
        turn = len(ids) - 1 - ids[::-1].index(self.im_start) if self.im_start in ids else len(ids)
        while pos < len(ids):
            # long contexts: smaller steps, so the step's attention scores stay within ATTN_BUDGET
            step = max(128, min(m.chunk, ATTN_BUDGET // ((pos + m.chunk) * self.q_heads * 2)))
            end = min(pos + step, len(ids), (pos // SNAP_EVERY + 1) * SNAP_EVERY)
            if pos < turn < end:
                end = turn
            logits = m.forward(ids[pos:end], last_only=True)
            mx.eval(logits, m.arrays(m.state))
            s.tokens.extend(ids[pos:end])
            s.pos = pos = end
            if pos % SNAP_EVERY == 0 or pos in (turn, len(ids)):
                self._snapshot(s)
            tick()
        return logits[-1]

    def generate(self, s, logits, max_tokens, sampling):
        """Yields generated token ids; stops at eos (not yielded) or max_tokens.
        Pipelined: the next token's forward pass is queued before the current token is handed back,
        so the GPU keeps working while the caller decodes text and writes to the socket."""
        from mlx_lm.sample_utils import make_sampler
        m = self.m
        if getattr(m, "head", None) is not None:
            m.state, m.pos = s.state, s.pos
            gen = m.generate_pipe if m.k == 1 else m.generate_spec
            yield from gen(s, logits, max_tokens, sampling, self.eos)
            return
        sampler = make_sampler(temp=sampling["temperature"] or 0.0, top_p=sampling["top_p"] or 0.0,
                               top_k=sampling["top_k"] or 0)

        def pick(lg):
            lg = lg.astype(mx.float32)
            return sampler((lg - mx.logsumexp(lg, keepdims=True))[None])[0]

        y = pick(logits)
        mx.async_eval(y)
        for n in range(max_tokens):
            nxt = None
            if n + 1 < max_tokens:
                m.state, m.pos = s.state, s.pos
                nxt = pick(m.forward(y[None])[-1])
                mx.async_eval(nxt)
            t = y.item()
            if nxt is not None:  # the state now includes t (even an eos): keep tokens/pos in step with it
                s.tokens.append(t)
                s.pos += 1
            if t in self.eos:
                return
            yield t
            if nxt is None:
                return
            y = nxt


# ============================================================ HTTP

class Disconnected(Exception):
    pass


def make_handler(runner, args):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, kind, msg):
            self._json(code, {"type": "error", "error": {"type": kind, "message": msg}})

        def do_GET(self):
            if self.path.split("?")[0] in ("/v1/models", "/models"):
                self._json(200, {"data": [{"type": "model", "id": args.model_name, "display_name": args.model_name,
                                           "created_at": "2026-11-02T00:00:00Z"}], "has_more": False,
                                 "first_id": args.model_name, "last_id": args.model_name})
            else:
                self._json(200, {"status": "ok"})

        def do_HEAD(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except ValueError:
                return self._error(400, "invalid_request_error", "body is not JSON")
            try:
                if path.endswith("/messages/count_tokens"):
                    ids, _, _ = runner.render(req, args.thinking)
                    return self._json(200, {"input_tokens": len(ids)})
                if path.endswith("/messages"):
                    return self.messages(req)
            except Disconnected:
                return
            except Exception as e:  # report, don't kill the server
                import tempfile, traceback
                traceback.print_exc()
                dump = os.path.join(tempfile.gettempdir(), "big2flash-failed-request.json")
                json.dump(req, open(dump, "w"))
                print(f"failed request saved to {dump}", flush=True)
                runner.sessions.clear()  # a session may hold half-updated state
                err = {"type": "error", "error": {"type": "api_error", "message": f"{type(e).__name__}: {e}"}}
                try:
                    if getattr(self, "streaming", False):  # headers already sent: report in-stream
                        self.wfile.write(f"event: error\ndata: {json.dumps(err)}\n\n".encode())
                        return
                    return self._json(500, err)
                except OSError:
                    return
            self._error(404, "not_found_error", f"no route for {path}")

        def messages(self, req):
            ids, think, prefix = runner.render(req, args.thinking)
            if len(ids) > args.max_context:
                return self._error(400, "invalid_request_error",
                                   f"prompt is too long: {len(ids)} tokens > {args.max_context} maximum")
            stream = bool(req.get("stream"))
            schemas = {t["name"]: t.get("input_schema", {}) for t in req.get("tools") or []}
            sampling = {k: req.get(k, runner.defaults[k]) for k in ("temperature", "top_p", "top_k")}
            max_tokens = min(int(req.get("max_tokens") or 4096), args.max_context - len(ids))
            stops = req.get("stop_sequences") or []
            msg_id = "msg_" + uuid.uuid4().hex[:24]
            model = req.get("model") or args.model_name

            self.streaming = stream
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

            def send(event, data):
                if not stream:
                    return
                try:
                    self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    raise Disconnected()

            with runner.lock:
                t0 = time.perf_counter()
                s, reused = runner._pick(ids)
                usage = {"input_tokens": len(ids) - reused, "cache_read_input_tokens": reused,
                         "cache_creation_input_tokens": 0, "output_tokens": 0}
                send("message_start", {"type": "message_start", "message": {
                    "id": msg_id, "type": "message", "role": "assistant", "model": model, "content": [],
                    "stop_reason": None, "stop_sequence": None, "usage": usage}})
                logits = runner.prefill(s, ids, reused, lambda: send("ping", {"type": "ping"}))
                t1 = time.perf_counter()

                blocks, cur = [], None  # content blocks (non-stream result) / index of the open streamed block
                def close():
                    nonlocal cur
                    if cur is not None:
                        if blocks[cur]["type"] == "thinking":
                            send("content_block_delta", {"type": "content_block_delta", "index": cur,
                                                         "delta": {"type": "signature_delta", "signature": "local"}})
                        send("content_block_stop", {"type": "content_block_stop", "index": cur})
                        cur = None

                def handle(events):
                    nonlocal cur
                    for ev in events:
                        kind = ev[0]
                        if kind == "tool":
                            close()
                            b = {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:24], "name": ev[1], "input": ev[2]}
                            blocks.append(b)
                            i = len(blocks) - 1
                            send("content_block_start", {"type": "content_block_start", "index": i,
                                                         "content_block": {**b, "input": {}}})
                            send("content_block_delta", {"type": "content_block_delta", "index": i,
                                                         "delta": {"type": "input_json_delta", "partial_json": json.dumps(ev[2])}})
                            send("content_block_stop", {"type": "content_block_stop", "index": i})
                            continue
                        if cur is None or blocks[cur]["type"] != kind:
                            close()
                            blocks.append({"type": "thinking", "thinking": "", "signature": "local"} if kind == "thinking"
                                          else {"type": "text", "text": ""})
                            cur = len(blocks) - 1
                            send("content_block_start", {"type": "content_block_start", "index": cur,
                                                         "content_block": {**blocks[cur], **({"signature": ""} if kind == "thinking" else {})}})
                        blocks[cur][kind] += ev[1]
                        send("content_block_delta", {"type": "content_block_delta", "index": cur,
                                                     "delta": {"type": "thinking_delta", "thinking": ev[1]} if kind == "thinking"
                                                     else {"type": "text_delta", "text": ev[1]}})

                parser = OutputParser(schemas, think, prefix)
                out, text, stop_reason, stop_seq = [], "", "end_turn", None
                # incremental detokenizing: decode only the tokens since the last emitted text (a full
                # re-decode per token costs ~1 ms at 8K output tokens and keeps growing)
                p = r = 0
                longest = max((len(x) for x in stops), default=0)
                for t in runner.generate(s, logits, max_tokens, sampling):
                    out.append(t)
                    full = runner.tok.decode(out[p:])
                    if full.endswith("�"):  # incomplete UTF-8 sequence: wait for the next token
                        continue
                    new = full[len(runner.tok.decode(out[p:r])):]
                    p, r = r, len(out)
                    text += new
                    tail = text[-(len(new) + longest):]
                    hit = next((x for x in stops if x in tail), None)
                    if hit:
                        new = new[:len(new) - (len(text) - text.index(hit, len(text) - len(tail)))]
                        stop_reason, stop_seq = "stop_sequence", hit
                        handle(parser.feed(new, final=True))
                        break
                    handle(parser.feed(new))
                else:
                    if len(out) >= max_tokens:
                        stop_reason = "max_tokens"
                    handle(parser.feed(runner.tok.decode(out[p:])[len(runner.tok.decode(out[p:r])):], final=True))
                close()
                if any(b["type"] == "tool_use" for b in blocks) and stop_reason == "end_turn":
                    stop_reason = "tool_use"
                usage["output_tokens"] = len(out)
                t2 = time.perf_counter()
                print(f"[{time.strftime('%H:%M:%S')}] prompt {len(ids)} (cached {reused}, prefill "
                      f"{len(ids) - reused} in {t1 - t0:.1f}s) -> {len(out)} tokens in {t2 - t1:.1f}s "
                      f"({len(out) / max(t2 - t1, 1e-9):.1f} tok/s) {stop_reason}"
                      f"{' tools=' + ','.join(b['name'] for b in blocks if b['type'] == 'tool_use') if any(b['type'] == 'tool_use' for b in blocks) else ''}",
                      flush=True)

            if stream:
                send("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": stop_seq},
                                       "usage": {"output_tokens": len(out)}})
                send("message_stop", {"type": "message_stop"})
            else:
                self._json(200, {"id": msg_id, "type": "message", "role": "assistant", "model": model,
                                 "content": blocks, "stop_reason": stop_reason, "stop_sequence": stop_seq,
                                 "usage": usage})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model-name", default="big2flash")
    ap.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                    help="auto: think when the request asks for it (Claude Code's thinking toggle)")
    ap.add_argument("--max-context", type=int, default=98304)
    ap.add_argument("--sessions", type=int, default=1, help="conversations kept cached at once")
    ap.add_argument("--wired-margin-gb", type=float, default=2.0, help="memory wired beyond the weights")
    ap.add_argument("--cache-gb", type=float, default=0.5, help="MLX buffer cache limit")
    ap.add_argument("--snapshots", type=int, default=4, help="rollback points kept per session")
    ap.add_argument("--mtp", type=int, default=1, help="tokens drafted per step by the model's MTP head "
                    "when it has one (0: off); 1 measured fastest on an M2 Max")
    args = ap.parse_args()
    runner = Runner(args.model, args.sessions, args.max_context, args.snapshots, args.wired_margin_gb, args.cache_gb,
                    args.mtp)
    args.max_context = runner.max_context
    # Single-threaded on purpose: MLX streams belong to the thread that made them (the model was
    # loaded on this one), and the GPU runs one request at a time anyway; others wait in the backlog.
    srv = HTTPServer((args.host, args.port), make_handler(runner, args))
    srv.request_queue_size = 64
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
