"""MTP speculative decoding vs plain decoding: identical greedy output? acceptance? speed?

usage: mtp_test.py MODEL_DIR [N_TOKENS]
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import mtp
from quiet import wait_quiet

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
model, tok = load(path)
mx.eval(model.parameters())
head = mtp.load_mtp(path, model.language_model)
PROMPTS = [
    "写一篇关于灯塔看守人的短篇小说，大约600字。",
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "解释一下 TCP 三次握手和四次挥手的过程，以及为什么需要 TIME_WAIT。",
]


def chat(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=True)


def plain(ids):
    cache = make_prompt_cache(model)
    y = mx.argmax(model(mx.array(ids)[None], cache=cache)[0, -1])
    mx.eval(y)
    out, t = [], time.perf_counter()
    for _ in range(N):
        nxt = mx.argmax(model(y.reshape(1, 1), cache=cache)[0, -1])
        mx.async_eval(nxt)
        out.append(y.item())
        y = nxt
    return out, N / (time.perf_counter() - t)


def spec(ids, k, temp=0.0):
    cache = make_prompt_cache(model)
    s = mtp.Speculator(model, head, k=k)
    lg, h = s.target(mx.array(ids)[None], cache)
    y = mx.argmax(lg[0, -1]).item()
    s.start(h, mx.array(ids), y)
    mx.eval(s.mtp_last)
    t = time.perf_counter()
    out = list(s.generate(cache, h, lg, y, N, temp=temp, top_p=0.95, top_k=20))
    return out, N / (time.perf_counter() - t), s.accepted / max(s.drafted, 1)


for p in PROMPTS:
    ids = chat(p)
    wait_quiet()
    base, v0 = plain(ids)
    line = f"{p[:24]:<26} plain {v0:5.1f} tok/s |"
    for k in (1, 2, 3, 4):
        wait_quiet()
        out, v, acc = spec(ids, k)
        same = out == base
        first_diff = next((i for i, (x, y) in enumerate(zip(out, base)) if x != y), None)
        line += f" k={k} {v:5.1f} tok/s acc {acc:.0%} {'same' if same else f'DIFF@{first_diff}'} |"
    print(line, flush=True)
wait_quiet()
out, v, acc = spec(chat(PROMPTS[0]), 3, temp=0.7)
print(f"sampled (temp 0.7, k=3): {v:.1f} tok/s acc {acc:.0%}")
print(tok.decode(out[:80]))
