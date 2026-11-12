"""Reference numbers from the stock mlx-lm generator (same model, same prompts as the server speed test)."""
import sys, mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
model, tok = load(sys.argv[1])
mx.set_wired_limit(mx.get_active_memory() + 2 * 2**30)
code = open("engine/server.py").read()[:30000]
for name, p in [("short", "Write a short story about a lighthouse keeper."),
                ("7.5K", code + "\n\nSummarize what this code does.")]:
    msgs = [{"role": "user", "content": p}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    for r in stream_generate(model, tok, prompt, max_tokens=200, sampler=make_sampler(0.0)):
        pass
    print(f"{name:5s}: prompt {r.prompt_tokens} @ {r.prompt_tps:.0f} tok/s | decode {r.generation_tokens} @ {r.generation_tps:.1f} tok/s | peak {r.peak_memory:.1f} GB", flush=True)
