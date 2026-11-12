"""Where an MTP round's time goes: acceptance, and the cost of each piece measured on its own."""
import sys, os, time, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx
from quiet import wait_quiet
import server, mtp
from transformers import AutoTokenizer

M = os.path.expanduser("~/Library/Application Support/big2flash/models/Qwen3.8-27B-4bit-MTP-fp16")
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
b = server.MlxLmBackend(M, 1)
tok = AutoTokenizer.from_pretrained(M)
emb, lm_head = b.lm.model.embed_tokens, b.lm.lm_head

def prompt(text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def timeit(f, n=30):
    f(); mx.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    mx.synchronize()
    return (time.perf_counter() - t) / n * 1000

for name, text in [("story", "写一篇关于灯塔看守人的短篇小说。"), ("code", "用 Python 写一个 LRU 缓存类，带测试。")]:
    ids = tok(prompt(text), add_special_tokens=False)["input_ids"]
    b.reset()
    lg = b.forward(ids, last_only=True)[-1]
    mx.eval(lg)
    s = types.SimpleNamespace(tokens=list(ids), pos=len(ids))
    rounds = [0]
    orig = mtp.rollback
    def rb(c, keep, total):
        rounds[0] += 1; rounds.append(keep)
        return orig(c, keep, total)
    mtp.rollback = rb
    wait_quiet()
    t = time.perf_counter()
    out = list(b.generate_spec(s, lg, 256, {"temperature": 0, "top_p": 1, "top_k": 0}, set()))
    dt = time.perf_counter() - t
    mtp.rollback = orig
    r = rounds[0]
    print(f"{name}: {len(out)} tok in {dt:.2f}s = {len(out)/dt:.2f} tok/s, {r} rounds, "
          f"{len(out)/r:.2f} tok/round, accept {(len(out)/r-1)*100:.0f}%, {dt/r*1000:.1f} ms/round")

# component costs at the current cache state
cache = b.state[:b.n]; hc = b.state[b.n]
x1 = mx.array([[1000]]); x2 = mx.array([[1000, 1001]])
def plain():
    h = b.lm.model(x1, cache=cache); mx.eval(lm_head(h).argmax(-1))
    for c in cache:
        if hasattr(c, "trim"): c.trim(1)
def verify(capture):
    def f():
        mtp._CAPTURE = capture
        h = b.lm.model(x2, cache=cache)
        mtp._CAPTURE = False
        mx.eval(lm_head(h).argmax(-1))
        for c in cache:
            if hasattr(c, "trim"): c.trim(2)
            elif capture: c.hist = None
    return f
hid = mx.zeros((1, 1, b.lm.args.hidden_size), mx.float16)
def draft():
    m = b.head(emb(x1), hid, hc); mx.eval(lm_head(m).argmax(-1)); hc.trim(1)
def head_only():
    m = b.head(emb(x1), hid, hc); mx.eval(m); hc.trim(1)
wait_quiet()
print(f"plain decode step {timeit(plain):.1f} ms")
print(f"verify 2 tok, no capture {timeit(verify(False)):.1f} ms")
print(f"verify 2 tok, capture {timeit(verify(True)):.1f} ms")
print(f"draft (head+lm_head) {timeit(draft):.1f} ms, head only {timeit(head_only):.1f} ms")
