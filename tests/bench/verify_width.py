"""Full-model verify forward (GDN capture on) per width T, stock kernels vs engine/qmm8; plus the same
with capture off, to see what per-token state capture costs."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx
import server, mtp, qmm8
from quiet import wait_quiet
M = os.path.expanduser("~/Library/Application Support/big2flash/models/Qwen3.8-27B-4bit-MTP-fp16")
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
b = server.MlxLmBackend(M, 1)
ids = list(range(1000, 2800))
b.reset(); mx.eval(b.forward(ids, last_only=True))
cache = b.state[:b.n]
def t_fwd(T, cap):
    x = mx.array([[1000 + i for i in range(T)]])
    mtp._CAPTURE = cap
    for _ in range(2): mx.eval(b.lm.lm_head(b.lm.model(x, cache=cache)))
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(4): mx.eval(b.lm.lm_head(b.lm.model(x, cache=cache)))
        best = min(best, (time.perf_counter() - t) / 4)
    mtp._CAPTURE = False
    return best * 1000
wait_quiet()
for T in (1, 2, 3, 4, 5, 6, 8):
    s_cap, s_nc = t_fwd(T, True), t_fwd(T, False)
    qmm8.install(); q_cap, q_nc = t_fwd(T, True), t_fwd(T, False); nn = qmm8.nn; nn.QuantizedLinear.__call__ = qmm8._ql_orig
    print(f"T={T}  stock {s_cap:6.1f} (no capture {s_nc:6.1f})   qmm8 {q_cap:6.1f} (no capture {q_nc:6.1f}) ms", flush=True)
