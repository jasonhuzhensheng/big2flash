"""Sampled-mode MTP speed with the model's default sampling, vs greedy, alternating."""
import sys, os, time, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
sys.path.insert(0, os.path.dirname(__file__))
import json, mlx.core as mx
from quiet import wait_quiet
import server
from transformers import AutoTokenizer
M = os.path.expanduser("~/Library/Application Support/big2flash/models/Qwen3.8-27B-4bit-MTP-fp16")
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
gc = json.load(open(os.path.join(M, "generation_config.json")))
b = server.MlxLmBackend(M, 1)
tok = AutoTokenizer.from_pretrained(M)
real_eval = mx.eval
acc = {"eval": 0.0, "n": 0}
def timed_eval(*a):
    t = time.perf_counter(); real_eval(*a); acc["eval"] += time.perf_counter() - t; acc["n"] += 1
for text in ["写一篇关于灯塔看守人的短篇小说。", "用 Python 写一个 LRU 缓存类，带测试。"]:
    ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
              add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
    for (name, smp), fn in [(m, f) for m in [("greedy ", {"temperature": 0, "top_p": 1, "top_k": 0}),
                             ("sampled", {k: gc.get(k) for k in ("temperature", "top_p", "top_k")})]
                            for f in ("generate_spec", "generate_pipe")] * 2:
        b.reset(); lg = b.forward(ids, last_only=True)[-1]; real_eval(lg)
        s = types.SimpleNamespace(tokens=list(ids), pos=len(ids))
        wait_quiet(); mx.random.seed(0)
        acc.update(eval=0.0, n=0); mx.eval = timed_eval
        t = time.perf_counter()
        out = list(getattr(b, fn)(s, lg, 256, smp, set()))
        dt = time.perf_counter() - t; mx.eval = real_eval
        print(f"{text[:6]} {name} {fn[9:]:5s} {len(out)/dt:5.2f} tok/s", flush=True)
