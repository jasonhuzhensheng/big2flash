"""Decode speed vs context length: the server's Runner path vs a bare mlx-lm loop on the same cache."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../engine"))
import server as S

MODEL = sys.argv[1]
r = S.Runner(MODEL, sessions=1, max_context=98304, snaps=4)
ids = [1000 + (i * 7919) % 50000 for i in range(24576)]
s, _ = r._pick(ids[:2])
done = 0
for n in (1024, 4096, 8192, 16384, 20480, 24000):
    r.prefill(s, ids[:n], done, lambda: None)
    done = n
    pos0, toks0 = s.pos, len(s.tokens)
    logits = r.m.forward([ids[n]])[-1]; s.pos += 1; s.tokens.append(ids[n])
    mx.eval(logits)
    # 1) server path (greedy, .item() each step)
    t = time.perf_counter()
    for k, _ in zip(range(48), r.generate(s, logits, 48, {"temperature": 0, "top_p": 1, "top_k": 0})):
        pass
    srv = 48 / (time.perf_counter() - t)
    # 2) bare loop, same cache, async pipelined
    y = mx.array([[ids[n]]])
    t = time.perf_counter()
    for k in range(48):
        out = r.m.model(y, cache=r.m.state)
        y = out[:, -1].argmax(-1)[None]
        mx.async_eval(y)
    mx.eval(y)
    bare = 48 / (time.perf_counter() - t)
    # roll back to the prompt so the next round continues from n
    snap_pos = s.pos
    print(f"ctx {n:6d}: server path {srv:5.1f} tok/s | bare async {bare:5.1f} tok/s | "
          f"active {mx.get_active_memory()/2**30:5.1f} GiB  cache {mx.get_cache_memory()/2**30:4.1f} GiB", flush=True)
    for c in r.m.state:  # undo the bare loop's 48 tokens on KV caches (recurrent state just drifts; fine for timing)
        if hasattr(c, "offset"):
            c.offset -= 48
    r.m.pos = s.pos = done
    s.tokens = s.tokens[:done]
    s.snaps = [(p, st) for p, st in s.snaps if p <= done]
