"""Needle retrieval past the QSA budget (sparse block selection active) plus prefill/decode timing."""
import os, sys, time, random
import mlx.core as mx
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(__file__))
from flash_next import FlashNext
M = "/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw"
tok = AutoTokenizer.from_pretrained(M)
m = FlashNext(M)
random.seed(0)
topics = ["river", "mountain", "harbor", "orchard", "library", "railway", "market", "observatory", "glacier", "bridge"]
def filler(n):
    out = []
    for i in range(n):
        a, b = random.sample(topics, 2)
        out.append(f"Report {i}: the {a} near the {b} was inspected on day {random.randint(1, 365)} and found in "
                   f"{random.choice(['good', 'fair', 'poor'])} condition with {random.randint(2, 90)} workers present.")
    return " ".join(out)
for n_lines in (40, 120, 380, 800):  # the first is a warm-up (kernel compilation)
    half = n_lines // 2
    doc = filler(half) + " IMPORTANT: the vault access code is 58213-KESTREL. " + filler(n_lines - half)
    msgs = [{"role": "user", "content": doc + "\n\nWhat is the vault access code mentioned in the reports? Answer with the code only."}]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]
    m.reset()
    t0 = time.perf_counter()
    for s in range(0, len(ids), 512):
        lg = m.forward(ids[s:s + 512], last_only=True); mx.eval(lg)
    tp = time.perf_counter() - t0
    out = [int(mx.argmax(lg[-1]).item())]
    t0 = time.perf_counter()
    for _ in range(24):
        out.append(int(mx.argmax(m.forward([out[-1]])[-1]).item()))
        if out[-1] == m.cfg.eos: break
    td = time.perf_counter() - t0
    print(f"{len(ids):6d} prompt tokens: prefill {tp:6.1f} s ({len(ids)/tp:5.0f} tok/s), decode {(len(out)-1)/td:5.1f} tok/s, "
          f"answer: {tok.decode(out).strip()!r}", flush=True)
