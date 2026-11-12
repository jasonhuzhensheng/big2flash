"""Same workload against any local engine, measured the same way from the client over streaming HTTP.

usage: engines.py KIND URL MODEL [--runs N] [--out results.jsonl] [--label NAME]
  KIND: anthropic (big2flash) | openai (LM Studio, mlx_lm.server, llama-server) | ollama
Cases: "short" (story, 256 tokens out), "long" (~7.5K-token prompt, 200 out),
  "code" (write a module + tests, 400 out), "edit" (~2.5K-token class in, type-hinted class out, 400), greedy, no thinking.
TTFT = request -> first content token (includes prefill); decode = (tokens - 1) / (last - first token).
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from quiet import wait_quiet

HERE = os.path.dirname(__file__)
CODE = open(os.path.join(HERE, "..", "..", "engine", "server.py")).read()[:30000]
SRC = open(os.path.join(HERE, "fixture_server.py")).read()
CLS = SRC[SRC.index("class MlxLmBackend"):SRC.index("class Runner")]
CASES = [
    ("code", "Write a Python module implementing a thread-safe LRU cache with per-entry TTL, "
             "then pytest tests covering eviction, expiry and concurrency. Code only.", 400),
    ("edit", "```python\n" + CLS[:9000] + "\n```\n\nAdd type hints to every method signature of this class. "
             "Output the complete updated class, unchanged otherwise. Code only.", 400),
    ("short", "写一篇关于灯塔看守人的短篇小说。", 256),
    ("long", CODE + "\n\n用中文总结这段代码做了什么，分点说明。", 200),
]


def stream(req):
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data: "):
                line = line[6:]
            if line and line != "[DONE]":
                try:
                    yield json.loads(line)
                except ValueError:
                    pass


def run(kind, url, model, prompt, max_tokens):
    msgs = [{"role": "user", "content": prompt}]
    if kind == "anthropic":
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": True, "messages": msgs}
        path = "/v1/messages"
    elif kind == "openai":
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": True, "messages": msgs,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        path = "/v1/chat/completions"
    else:
        body = {"model": model, "messages": msgs, "stream": True, "think": False,
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 16384}}
        path = "/api/chat"
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    chunks, n_out, n_in = 0, None, None
    for ev in stream(req):
        text = ""
        if kind == "anthropic":
            if ev.get("type") == "message_start":
                u = ev["message"]["usage"]
                n_in = u["input_tokens"] + u.get("cache_read_input_tokens", 0)
            elif ev.get("type") == "content_block_delta":
                text = ev["delta"].get("text") or ev["delta"].get("thinking") or ""
            elif ev.get("type") == "message_delta":
                n_out = ev["usage"]["output_tokens"]
        elif kind == "openai":
            for c in ev.get("choices") or []:
                d = c.get("delta") or {}
                text += (d.get("content") or "") + (d.get("reasoning_content") or "")
            if ev.get("usage"):
                n_in, n_out = ev["usage"]["prompt_tokens"], ev["usage"]["completion_tokens"]
        else:
            text = (ev.get("message") or {}).get("content", "") + (ev.get("message") or {}).get("thinking", "")
            if ev.get("done"):
                n_in, n_out = ev.get("prompt_eval_count"), ev.get("eval_count")
        if text:
            now = time.perf_counter()
            first = first or now
            last = now
            chunks += 1
    n_out = n_out or chunks
    ttft = first - t0
    return {"prompt_tokens": n_in, "output_tokens": n_out, "ttft_s": round(ttft, 3),
            "prefill_tps": round(n_in / ttft, 1) if n_in else None,
            "decode_tps": round((n_out - 1) / (last - first), 2) if last > first else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", default="short,long")
    a = ap.parse_args()
    if a.model == "auto":  # the first model the server lists (OpenAI-style /v1/models)
        a.model = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=30))["data"][0]["id"]
    # warm-up: loads weights / compiles. Servers often answer (503, refused) while still loading
    t0 = time.time()
    while True:
        try:
            run(a.kind, a.url, a.model, "你好", 8)
            break
        except (OSError, urllib.error.HTTPError) as e:
            if time.time() - t0 > 900:
                raise
            time.sleep(5)
    for name, prompt, n in [c for c in CASES if c[0] in a.cases.split(",")]:
        best = None
        for i in range(a.runs):
            # vary the prompt per run so no engine can answer the long case from its prompt cache
            p = prompt if i == 0 else f"（第{i + 1}次）" + prompt
            wait_quiet()
            r = run(a.kind, a.url, a.model, p, n)
            if best is None or (r["decode_tps"] or 0) > (best["decode_tps"] or 0):
                best = r
        rec = {"engine": a.label or a.kind, "case": name, **best}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if a.out:
            open(a.out, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
