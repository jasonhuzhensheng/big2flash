"""End-to-end server check: the same conversation against --mtp 1 and --mtp 0 must give the same
greedy text. It covers a multi-turn conversation reusing the cache, a stop sequence that ends
generation early (the generator is closed mid-round), a client that disconnects mid-stream, an
edit of an earlier turn (rollback to a snapshot), and a sampled request.
usage: server_check.py [engine/server.py]"""
import json, os, socket, subprocess, sys, time, urllib.request

SERVER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "engine", "server.py")
PY = os.path.expanduser("~/Library/Application Support/big2flash/venv/bin/python")
M = os.path.expanduser("~/Library/Application Support/big2flash/models/Qwen3.8-27B-4bit-MTP-fp16")
PORT = 8098
CODE = open(os.path.join(os.path.dirname(__file__), "..", "engine", "mtp.py")).read()[:6000]


def ask(msgs, n=120, **kw):
    body = {"model": "big2flash", "max_tokens": n, "temperature": 0, "stream": True, "messages": msgs, **kw}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/messages", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    text, usage, stop = "", {}, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "message_start":
                usage = ev["message"]["usage"]
            elif ev["type"] == "content_block_delta":
                text += ev["delta"].get("text", "")
            elif ev["type"] == "message_delta":
                stop = ev["delta"].get("stop_reason")
    return text, usage.get("cache_read_input_tokens", 0), stop


def disconnect(msgs):
    """Start streaming, read a few events, drop the connection."""
    body = json.dumps({"model": "big2flash", "max_tokens": 300, "temperature": 0, "stream": True,
                       "messages": msgs}).encode()
    s = socket.create_connection(("127.0.0.1", PORT))
    s.sendall(f"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    got = b""
    while got.count(b"content_block_delta") < 15:
        got += s.recv(4096)
    s.close()
    time.sleep(1)


def session():
    out = []
    u1 = [{"role": "user", "content": CODE + "\n\n用中文解释这段代码的作用。"}]
    t1, c1, _ = ask(u1)
    out.append(("turn 1", t1, c1))
    u2 = u1 + [{"role": "assistant", "content": t1}, {"role": "user", "content": "rollback 函数具体做什么？"}]
    t2, c2, _ = ask(u2)
    out.append(("turn 2 (cache reuse)", t2, c2))
    t3, c3, stop = ask(u2 + [{"role": "assistant", "content": t2}, {"role": "user", "content": "列出三个要点，用 1. 2. 3. 编号。"}],
                       stop_sequences=["2."])
    out.append((f"turn 3 (stop sequence, {stop})", t3, c3))
    disconnect(u2 + [{"role": "assistant", "content": t2}, {"role": "user", "content": "写一首关于缓存的诗。"}])
    t4, c4, _ = ask(u2 + [{"role": "assistant", "content": t2}, {"role": "user", "content": "MTP 是什么的缩写？"}])
    out.append(("turn 4 (after a disconnect)", t4, c4))
    e1 = [{"role": "user", "content": CODE + "\n\n这段代码里 Holes 类是做什么的？"}]
    t5, c5, _ = ask(e1)
    out.append(("edited turn 1 (rollback)", t5, c5))
    t6, _, _ = ask(u1, n=60, temperature=1.0, top_p=0.95, top_k=20)
    out.append(("sampled", t6[:40].replace("\n", " ") + "…", 0))
    return out


results = {}
for mtp in ("1", "0"):
    log = open(f"/tmp/server_check_{mtp}.log", "w")
    p = subprocess.Popen([PY, "-u", SERVER, "--model", M, "--port", str(PORT), "--mtp", mtp],
                         stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=2); break
            except Exception:
                time.sleep(1)
        results[mtp] = session()
    finally:
        p.terminate(); p.wait()

ok = True
for (name, a, ca), (_, b, cb) in zip(results["1"], results["0"]):
    same = a == b or name == "sampled"
    ok &= same
    print(f"{name:32s} cached {ca:5d}/{cb:5d}  {'same' if same else 'DIFFERENT'}  {a[:50]!r}")
    if not same:
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"    mtp1 …{a[max(0, i-20):i+30]!r}\n    mtp0 …{b[max(0, i-20):i+30]!r}")
print("PASS" if ok else "CHECK DIFFERENCES")
