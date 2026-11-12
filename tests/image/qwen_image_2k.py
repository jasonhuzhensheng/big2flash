"""2048x2048 Qwen-Image-2.1-Turbo generation test, LocalImageStudio's verified recipe (8 steps, CFG 1, bf16, MPS)."""
import os, sys, time
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import torch
from diffusers import QwenImage21Pipeline

MODEL = "/Volumes/4T 001/AI模型/LocalImageStudio/internal-model-store/views/local-image-studio/Qwen-Image-2.1-Turbo"
size, out = int(sys.argv[1]), sys.argv[2]
t0 = time.monotonic()
pipe = QwenImage21Pipeline.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True, use_safetensors=True)
pipe.vae.enable_tiling(tile_sample_min_height=512, tile_sample_min_width=512,
                       tile_sample_stride_height=384, tile_sample_stride_width=384)
pipe.to("mps")
print(f"loaded in {time.monotonic() - t0:.0f}s", flush=True)
prev = t1 = time.monotonic()
def step(_p, i, _t, kw):
    global prev
    torch.mps.synchronize()
    now = time.monotonic()
    print(f"step {i + 1}/8 {now - prev:.1f}s", flush=True)
    prev = now
    return kw
img = pipe(prompt="A misty mountain lake at sunrise, a small wooden boat, pine forest, photorealistic, highly detailed",
           width=size, height=size, true_cfg_scale=1.0, use_kv_cache=True,
           generator=torch.Generator("cpu").manual_seed(42), callback_on_step_end=step).images[0]
torch.mps.synchronize()
print(f"denoise+decode {time.monotonic() - t1:.0f}s, total {time.monotonic() - t0:.0f}s, peak MPS "
      f"{torch.mps.driver_allocated_memory() / 2**30:.1f} GiB", flush=True)
img.save(out)
print("saved", out, img.size)
