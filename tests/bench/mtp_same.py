"""MTP generation must match plain greedy decoding, and leave a state that continues like it.
generate_spec is bit-identical. generate_pipe attends over masked slots, so float sums may round
differently: it may only diverge where plain decoding's top two logits are within TIE of each other.
Each case also stops at an odd length and continues with plain decoding from the state left behind
(checks cache compaction, linear-attention state and the head's pending state)."""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
import mlx.core as mx
import server
from transformers import AutoTokenizer
M = os.path.expanduser("~/Library/Application Support/big2flash/models/Qwen3.8-27B-4bit-MTP-fp16")
TIE = 0.05
b = server.MlxLmBackend(M, 1)
tok = AutoTokenizer.from_pretrained(M)
GREEDY = {"temperature": 0, "top_p": 1, "top_k": 0}

def plain_run(lg, n):
    out, gaps = [], []
    for _ in range(n):
        l = lg.astype(mx.float32); t = mx.argmax(l).item(); top = mx.sort(l)[-2:].tolist()
        out.append(t); gaps.append(top[1] - top[0]); lg = b.forward([t], last_only=True)[-1]
    return out, gaps

def check(a, ref, gaps):
    i = next((i for i, (x, y) in enumerate(zip(a, ref)) if x != y), None)
    if i is None:
        return "identical", True
    return f"diverges at {i} (gap {gaps[i]:.4f})", gaps[i] < TIE

ok = True
for text in ["写一篇关于灯塔看守人的短篇小说。", "用 Python 写一个 LRU 缓存类，带测试。", "Explain TCP slow start briefly."]:
    ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
              add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
    b.reset(); ref, gaps = plain_run(b.forward(ids, last_only=True)[-1], 240)
    for name in ("generate_spec", "generate_pipe"):
        for stop in (200, 157):
            b.reset(); lg = b.forward(ids, last_only=True)[-1]
            s = types.SimpleNamespace(tokens=list(ids), pos=len(ids))
            out = list(getattr(b, name)(s, lg, stop, GREEDY, set()))
            st = s.tokens == list(ids) + out and s.pos == len(s.tokens) and len(out) == stop
            # the state has processed all of out; plain decoding's next token is ref[stop]
            cont_ok = True
            if out == ref[:stop]:
                c = ref[stop]
                lgc = b.forward([c], last_only=True)[-1]
                more, _ = plain_run(lgc, 30)
                cont_ok = more == ref[stop + 1:stop + 31]
            msg, good = check(out, ref, gaps)
            good = good and st and cont_ok
            ok &= good
            print(f"{text[:10]} {name} stop={stop}: {msg}, state {'ok' if st else 'BAD'}, "
                  f"continuation {'ok' if cont_ok else 'BAD'}" + ("" if out == ref[:stop] else " (not checked)"), flush=True)
print("PASS" if ok else "FAIL")
