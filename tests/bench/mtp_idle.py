"""Per MTP round: time inside mx.eval (GPU work + wait) vs everything else (Python, graph building)."""
import sys, os, time, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx
from quiet import wait_quiet
import server
from transformers import AutoTokenizer
M = os.path.expanduser("~/Library/Application Support/big2flash/models/Qwen3.8-27B-4bit-MTP-fp16")
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
b = server.MlxLmBackend(M, 1)
tok = AutoTokenizer.from_pretrained(M)
ids = tok(tok.apply_chat_template([{"role": "user", "content": "写一篇关于灯塔看守人的短篇小说。"}], tokenize=False,
          add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
real_eval = mx.eval
acc = {"eval": 0.0, "n": 0}
def timed_eval(*a):
    t = time.perf_counter(); real_eval(*a); acc["eval"] += time.perf_counter() - t; acc["n"] += 1
import mtp
for run in range(6):
    mtp._COMPILE = run % 2 == 1
    b.reset(); lg = b.forward(ids, last_only=True)[-1]; real_eval(lg)
    s = types.SimpleNamespace(tokens=list(ids), pos=len(ids))
    wait_quiet()
    acc.update(eval=0.0, n=0); mx.eval = timed_eval
    t = time.perf_counter()
    out = list(b.generate_spec(s, lg, 256, {"temperature": 0, "top_p": 1, "top_k": 0}, set()))
    dt = time.perf_counter() - t; mx.eval = real_eval
    r = acc["n"]
    print(("compiled " if mtp._COMPILE else "plain    ") + f"{len(out)/dt:.2f} tok/s, {r} rounds: {dt/r*1000:.1f} ms/round, in eval {acc['eval']/r*1000:.1f}, "
          f"outside {(dt-acc['eval'])/r*1000:.1f}")

