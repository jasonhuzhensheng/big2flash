"""mtp.accept_sampled must emit exactly the target's distribution p0 whatever the draft q."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
import mlx.core as mx
import mtp
mx.random.seed(0)
V, N = 6, 100000
p0 = mx.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
p1 = mx.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
worst = 0
for q in [mx.array([0.05, 0.05, 0.10, 0.20, 0.30, 0.30]), p0, mx.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])]:
    def one(key):
        d = mtp.categorical(q)
        acc, nxt = mtp.accept_sampled(p0, p1, d, q)
        return mx.where(acc, d, nxt), acc
    first, accs = [], []
    for i in range(N // 1000):
        toks = [one(None) for _ in range(1000)]
        mx.eval(toks)
        first += [int(t.item()) for t, _ in toks]; accs += [bool(a.item()) for _, a in toks]
    freq = [first.count(v) / len(first) for v in range(V)]
    err = max(abs(f - p) for f, p in zip(freq, p0.tolist()))
    worst = max(worst, err)
    print("q", [round(x, 2) for x in q.tolist()], "freq", [round(f, 3) for f in freq],
          f"max err {err:.4f}, accept {sum(accs)/len(accs):.3f} (theory {mx.minimum(p0, q).sum().item():.3f})")
print("EXACT" if worst < 0.006 else "WRONG")
