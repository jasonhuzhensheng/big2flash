"""Coding workloads only: MTP draft depth k=1..3 (and the k=1 pipeline), verify cost per width,
and how often copying from the context (prompt lookup) would guess the next tokens."""
import sys, os, time, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx
from quiet import wait_quiet
import server, mtp
from transformers import AutoTokenizer

HERE = os.path.dirname(__file__)
M = os.path.expanduser("~/Library/Application Support/big2flash/models/Qwen3.8-27B-4bit-MTP-fp16")
SRC = open(os.path.join(HERE, "fixture_server.py")).read()
CLS = SRC[SRC.index("class MlxLmBackend"):SRC.index("class Runner")] if "class Runner" in SRC else SRC[:12000]
CASES = {
    "new": "Write a Python module implementing a thread-safe LRU cache with per-entry TTL, "
           "then pytest tests covering eviction, expiry and concurrency. Code only.",
    "edit": "```python\n" + CLS[:9000] + "\n```\n\nAdd type hints to every method signature of this class. "
            "Output the complete updated class, unchanged otherwise. Code only.",
}
N = 400
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
b = server.MlxLmBackend(M, 1)
tok = AutoTokenizer.from_pretrained(M)
GREEDY = {"temperature": 0, "top_p": 1, "top_k": 0}

rounds = []
orig_rb = mtp.rollback
def rb(c, keep, total):
    rounds.append(keep)
    return orig_rb(c, keep, total)
mtp.rollback = rb


def ids_of(text):
    return tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
               add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]


def run(ids, fn, k):
    b.k = k
    b.reset(); lg = b.forward(ids, last_only=True)[-1]; mx.eval(lg)
    s = types.SimpleNamespace(tokens=list(ids), pos=len(ids))
    wait_quiet()
    rounds.clear()
    t = time.perf_counter()
    out = list(getattr(b, fn)(s, lg, N, GREEDY, set()))
    return out, len(out) / (time.perf_counter() - t), list(rounds)


def lookup_stats(ctx, out, L=8, nmin=2, nmax=4):
    """Offline: at each step, find the latest earlier occurrence of the last n tokens (n=nmax..nmin)
    in prompt+output so far, propose the L tokens after it, count how many match the real output.
    Simulates rounds: a hit advances by matched+1 tokens, a miss (or no match) by 1 MTP round."""
    seq, i, rounds_, hits, gained = ctx + out, 0, 0, 0, 0
    P = len(ctx)
    while i < len(out):
        pos = P + i
        best = None
        for n in range(nmax, nmin - 1, -1):
            pat = seq[pos - n:pos]
            for j in range(pos - n - 1, -1, -1):
                if seq[j:j + n] == pat:
                    best = j + n
                    break
            if best is not None:
                break
        m = 0
        if best is not None:
            prop = seq[best:best + L]
            while m < len(prop) and i + m < len(out) and prop[m] == out[i + m]:
                m += 1
        rounds_ += 1
        hits += m > 0
        gained += m
        i += m + 1
    return {"rounds": rounds_, "tok_per_round": round(len(out) / rounds_, 2), "hit_rate": round(hits / rounds_, 2)}


# verify cost per width T, at a realistic context length
ids = ids_of(CASES["edit"])
b.reset(); mx.eval(b.forward(ids, last_only=True))
cache = b.state[:b.n]
wait_quiet()
print(f"context {len(ids)} tokens; verify forward (with GDN capture) per width:")
for T in (1, 2, 3, 4, 6, 8, 12, 16):
    x = mx.array([[1000 + i for i in range(T)]])
    mtp._CAPTURE = True
    for _ in range(2):
        mx.eval(b.lm.lm_head(b.lm.model(x, cache=cache)))
    t = time.perf_counter()
    for _ in range(10):
        mx.eval(b.lm.lm_head(b.lm.model(x, cache=cache)))
    mtp._CAPTURE = False
    print(f"  T={T:2d} {(time.perf_counter() - t) / 10 * 1000:6.1f} ms", flush=True)

for name, text in CASES.items():
    ids = ids_of(text)
    print(f"\n== {name} (prompt {len(ids)} tokens)", flush=True)
    ref = None
    for rep in range(2):
        for fn, k in [("generate_pipe", 1), ("generate_spec", 1), ("generate_spec", 2), ("generate_spec", 3)]:
            out, tps, rs = run(ids, fn, k)
            if ref is None:
                ref = out
            same = "same" if out == ref else f"DIFF@{next((i for i, (a, c) in enumerate(zip(out, ref)) if a != c), min(len(out), len(ref)))}"
            dist = [rs.count(j) / len(rs) for j in range(1, k + 2)] if rs else []
            print(f"  {fn[9:]:5s} k={k} {tps:6.2f} tok/s  tok/round {len(out) / max(len(rs), 1):.2f}  "
                  f"keep-dist {[round(v, 2) for v in dist]}  {same}", flush=True)
    for L in (4, 8, 16):
        print(f"  prompt-lookup L={L}: {lookup_stats(ids, ref, L)}")
    print("  tail:", repr(tok.decode(ref[-60:]))[:200])
