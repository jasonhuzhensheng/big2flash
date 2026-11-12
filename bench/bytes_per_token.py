"""Exact bytes a Qwen3.8-Flash-Next decode step must read, from the safetensors headers."""
import collections, glob, json, os, re, struct, sys

model = sys.argv[1]
cfg = json.load(open(os.path.join(model, "config.json")))
tc = cfg.get("text_config", cfg)
E, K = tc["num_experts"], tc["num_experts_per_tok"]

sizes = {}
for f in glob.glob(os.path.join(model, "*.safetensors")):
    with open(f, "rb") as h:
        hd = json.loads(h.read(struct.unpack("<Q", h.read(8))[0]))
    for k, v in hd.items():
        if k != "__metadata__":
            sizes[k] = v["data_offsets"][1] - v["data_offsets"][0]

read = collections.Counter()   # bytes read per decoded token
stored = collections.Counter()
for k, s in sizes.items():
    if "vision" in k or "visual" in k:
        cat, frac = "vision (not read)", 0
    elif k.startswith("language_model.mtp"):
        cat, frac = "mtp head (per draft)", 1
    elif "switch_mlp" in k:
        cat, frac = "routed experts", K / E
    elif "embed_tokens" in k:
        cat, frac = "embedding (1 row)", 0
    elif "lm_head" in k:
        cat, frac = "lm_head", 1
    elif "shared_expert" in k:
        cat, frac = "shared expert", 1
    elif "linear_attn" in k:
        cat, frac = "linear attention", 1
    elif "self_attn" in k:
        cat, frac = "full attention + indexer", 1
    elif "hyper_connection" in k:
        cat, frac = "hyper-connections", 1
    elif ".ple." in k:
        cat, frac = "PLE projections", 1
    elif "mlp.gate" in k:
        cat, frac = "router", 1
    else:
        cat, frac = "other", 1
    stored[cat] += s
    read[cat] += s * frac

G = 1e9
print(f"{'component':28s} {'stored GB':>10s} {'read/token GB':>14s}")
for cat in sorted(stored, key=lambda c: -read[c]):
    print(f"{cat:28s} {stored[cat]/G:10.3f} {read[cat]/G:14.4f}")
main = sum(v for c, v in read.items() if not c.startswith("mtp"))
print(f"{'TOTAL (main model)':28s} {sum(stored.values())/G:10.3f} {main/G:14.4f}")
json.dump({"read_per_token_bytes": main, "by_component": read, "mtp_bytes": read["mtp head (per draft)"],
           "expert_bytes_each": stored["routed experts"] / (E * tc["num_hidden_layers"])},
          open(os.path.join(os.path.dirname(__file__), "bytes_per_token.json"), "w"), indent=1)
