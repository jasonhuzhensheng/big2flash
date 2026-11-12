"""Load Flash-Next, check it (perplexity on a fixed passage, a greedy chat answer), time decode."""
import argparse, math, os, sys, time
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from flash_next import FlashNext

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/Volumes/4T 001/mlx-models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw")
ap.add_argument("--prompt", default="用三句话解释为什么天空是蓝色的。")
ap.add_argument("--max-tokens", type=int, default=120)
ap.add_argument("--chunk", type=int, default=512)
args = ap.parse_args()

PASSAGE = ("The Industrial Revolution began in Great Britain in the late eighteenth century and spread to "
           "continental Europe and North America over the following decades. It marked a shift from hand "
           "production to machines, new chemical manufacturing and iron production processes, the increasing "
           "use of steam power and water power, the development of machine tools and the rise of the mechanised "
           "factory system. Output greatly increased, and the result was an unprecedented rise in population and "
           "in the rate of population growth. The textile industry was the first to use modern production "
           "methods, and textiles became the dominant industry in terms of employment, value of output and "
           "capital invested.")

tok = AutoTokenizer.from_pretrained(args.model)
t0 = time.perf_counter()
m = FlashNext(args.model)
print(f"loaded in {time.perf_counter() - t0:.1f} s, active memory {mx.get_active_memory() / 2**30:.1f} GiB", flush=True)


def prefill(ids):
    logits = None
    for s in range(0, len(ids), args.chunk):
        logits = m.forward(ids[s:s + args.chunk], last_only=True)
        mx.eval(logits, [st for st in m.state for st in st.values() if isinstance(st, mx.array)])
    return logits


# perplexity of the passage (teacher forcing)
ids = tok(PASSAGE)["input_ids"]
m.reset()
logits = m.forward(ids)
lp = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
nll = -mx.take_along_axis(lp, mx.array(ids[1:])[:, None], axis=-1).mean().item()
print(f"passage perplexity: {math.exp(nll):.3f} over {len(ids) - 1} tokens", flush=True)

# greedy chat answer with timing
msgs = [{"role": "user", "content": args.prompt}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text)["input_ids"]
m.reset()
t0 = time.perf_counter()
logits = prefill(ids)
nxt = int(mx.argmax(logits[-1]).item())
t_prefill = time.perf_counter() - t0
out = [nxt]
t0 = time.perf_counter()
for _ in range(args.max_tokens - 1):
    logits = m.forward([nxt])
    nxt = int(mx.argmax(logits[-1]).item())
    out.append(nxt)
    if nxt in (tok.eos_token_id, m.cfg.eos):
        break
t_dec = time.perf_counter() - t0
print("answer:", tok.decode(out))
print(f"prefill {len(ids)} tokens in {t_prefill:.2f} s ({len(ids) / t_prefill:.0f} tok/s); "
      f"decode {len(out) - 1} tokens in {t_dec:.2f} s ({(len(out) - 1) / t_dec:.1f} tok/s)")
