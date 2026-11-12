"""Looks at this Mac and the local models, and recommends the fastest setup that fits.

  python hwprobe.py            # JSON on stdout, no GPU work (safe to run any time)
  python hwprobe.py --measure  # also times a real weight read on the GPU (~5 s) to replace the table guess

Decode speed of a dense model is set by memory bandwidth: each token reads every weight once.
So the estimate is  bandwidth x efficiency / weight bytes, times the MTP gain when the model has an MTP head.
Calibrated on M2 Max 96GB: nominal 400 GB/s, measured 355, plain decode 22.2 tok/s on 15.1 GB, MTP x1.1-1.25.
"""
import json, os, re, subprocess, sys
from pathlib import Path

# Apple's published unified-memory bandwidth, GB/s. Keyed by (generation, tier); some tiers depend on GPU cores.
NOMINAL = {
    ("M1", ""): 68, ("M1", "Pro"): 200, ("M1", "Max"): 400, ("M1", "Ultra"): 800,
    ("M2", ""): 100, ("M2", "Pro"): 200, ("M2", "Max"): 400, ("M2", "Ultra"): 800,
    ("M3", ""): 100, ("M3", "Pro"): 150, ("M3", "Max"): (300, 400, 40), ("M3", "Ultra"): 819,
    ("M4", ""): 120, ("M4", "Pro"): 273, ("M4", "Max"): (410, 546, 40),
}
EFF = 0.83        # plain decode tok/s x weight GB / nominal GB/s, measured here
MEASURED_EFF = 0.94  # same, against the --measure number
MTP_GAIN = 1.15   # conservative: prose ~1.1, code ~1.25
MODELS = Path.home() / "Library/Application Support/big2flash/models"


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def chip():
    brand = sh("sysctl", "-n", "machdep.cpu.brand_string")
    m = re.match(r"Apple (M\d+)\s*(Pro|Max|Ultra)?", brand)
    gen, tier = (m.group(1), m.group(2) or "") if m else ("?", "")
    gpu = re.search(r"Total Number of Cores:\s*(\d+)", sh("system_profiler", "SPDisplaysDataType"))
    cores = int(gpu.group(1)) if gpu else 0
    bw = NOMINAL.get((gen, tier))
    if isinstance(bw, tuple):  # (binned, full, full-core-count)
        bw = bw[1] if cores >= bw[2] else bw[0]
    return {
        "brand": brand, "model": sh("sysctl", "-n", "hw.model"), "generation": gen, "tier": tier,
        "gpu_cores": cores, "memory_gb": int(sh("sysctl", "-n", "hw.memsize") or 0) / 2**30,
        "p_cores": int(sh("sysctl", "-n", "hw.perflevel0.physicalcpu") or 0),
        "macos": sh("sw_vers", "-productVersion"), "nominal_gbps": bw,
    }


def measure_gbps():
    """Reads ~4 GB of 4-bit weights through quantized matmul, the same path decode uses."""
    import time, mlx.core as mx
    n, k, layers = 8192, 8192, 120  # 120 x 32 MB packed = ~4 GB, larger than any cache
    w = [mx.quantize(mx.random.normal((n, k), dtype=mx.float16), group_size=64, bits=4) for _ in range(layers)]
    mx.eval(w)
    x = mx.random.normal((1, k), dtype=mx.float16)
    nbytes = sum(a.nbytes for q in w for a in q)
    best = 0.0
    for _ in range(5):
        h = x
        t = time.perf_counter()
        for q, s, b in w:  # dependent chain, like layers
            h = mx.quantized_matmul(h, q, s, b, transpose=True, group_size=64, bits=4)[:, :k] * 1e-3
        mx.eval(h)
        best = max(best, nbytes / (time.perf_counter() - t) / 1e9)
    return round(best, 1)


def models():
    out = []
    for d in sorted(MODELS.glob("*")) if MODELS.exists() else []:
        if not (d / "config.json").exists():
            continue
        # Decode reads the language weights only: leave out vision towers and draft heads.
        body = sum(f.stat().st_size for f in d.glob("model*.safetensors") if "vision" not in f.name)
        cfg = json.loads((d / "config.json").read_text())
        t = cfg.get("text_config", cfg)
        q = cfg.get("quantization") or cfg.get("quantization_config") or {}
        if (d / "mtp/model.safetensors").exists():
            draft = "mtp"                   # big2flash layout
        elif (d / "mtp.safetensors").exists():
            draft = "mtp-mtplx"             # same head, MTPLX's file layout
        elif (d / "drafter").is_dir():
            draft = "dflash"                # separate drafter model (mlx-serve)
        else:
            draft = None
        out.append({"name": d.name, "path": str(d), "weights_gb": round(body / 1e9, 1),
                    "bits": q.get("bits"), "group_size": q.get("group_size"),
                    "size_class": (t.get("hidden_size"), t.get("num_hidden_layers")),
                    "draft": draft, "mtp": draft == "mtp", "big2flash": draft in ("mtp", None)})
    return out


def recommend(hw, found, gbps=None):
    mem = hw["memory_gb"]
    # GPU may wire about 2/3 of RAM on small Macs, 3/4 on 36 GB and up; leave room for a 32K-token KV cache.
    budget = mem * (0.75 if mem >= 36 else 0.66) - 4
    if gbps:
        speed = gbps * MEASURED_EFF
    elif hw["nominal_gbps"]:
        speed = hw["nominal_gbps"] * EFF
    else:
        speed = None
    ranked = []
    for m in found:
        fits = m["weights_gb"] <= budget
        est = round(speed / m["weights_gb"] * (MTP_GAIN if m["mtp"] else 1), 1) if speed and m["weights_gb"] else None
        ranked.append({**m, "fits": fits, "est_tok_s": est})
    # Biggest model class that fits (quality), then the fastest pack of it that big2flash runs.
    usable = [m for m in ranked if m["fits"] and m["big2flash"]]
    top = max((m["size_class"] for m in usable), key=lambda c: (c[0] or 0) * (c[1] or 0), default=None)
    usable = sorted((m for m in usable if m["size_class"] == top), key=lambda m: -(m["est_tok_s"] or 0))
    best = usable[0] if usable else None
    config = {
        "mtp": bool(best and best["mtp"]),
        "wired": mem >= 24,                     # wiring weights is what takes reads from ~40 to ~355 GB/s here
        "compile": True,
        "prefill_chunk": 2048 if mem >= 64 else 512,
        "max_context": 32768 if budget - (best or {}).get("weights_gb", 0) > 8 else 8192,
    }
    notes = []
    if not usable:
        notes.append(f"本机可给 GPU 约 {budget:.0f} GB，本地没有放得下的模型；需要更小的量化版本。")
    elif not best["mtp"]:
        notes.append("推荐的模型没有 MTP 头，无法用推测解码；换带 MTP 的版本约快 15%。")
    return {"budget_gb": round(budget, 1), "best": best, "config": config, "models": ranked, "notes": notes}


if __name__ == "__main__":
    hw = chip()
    gbps = measure_gbps() if "--measure" in sys.argv else None
    hw["measured_gbps"] = gbps
    print(json.dumps({"hardware": hw, **recommend(hw, models(), gbps)}, ensure_ascii=False, indent=2))
