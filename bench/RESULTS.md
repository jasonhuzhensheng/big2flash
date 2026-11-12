# Step 0: M2 Max roofline and baseline (2026-11-02)

Machine: Apple M2 Max, 38-core GPU, 96 GB. MLX 0.32.4, mlx-lm 0.32.2.
GPU working set allowed by default: 83.5 GB (enough for Flash-Next's 70 GiB of weights).

## Hardware ceilings (measured)
| Probe | Result |
|---|---|
| Read bandwidth (sum over 2 GiB) | 338-365 GB/s (spec 400) |
| Per-kernel dispatch overhead | 4.6-4.9 us |

## Flash-Next 4.7bpw artifact: bytes per decoded token (from safetensors headers)
Dense layers are 8-bit (8.5 bpw), routed experts 4-bit (4.5 bpw).
| Component | GB/token |
|---|---|
| Linear attention (36 layers) | 2.218 |
| Routed experts (10 of 512 x 48) | 1.327 |
| Hyper-connections | 0.685 |
| lm_head | 0.675 |
| Full attention + indexer (12 layers) | 0.656 |
| Shared expert | 0.251 |
| Router + PLE | 0.102 |
| **Total** | **5.915** |
| MTP head, per draft token | 1.512 |

Decode roofline at 365 GB/s: 16.2 ms/token = **62 tok/s** (no speculation).
With dense layers re-quantised to 4-bit: 3.73 GB/token = **98 tok/s** (changes the output).

## Stock MLX efficiency
| Kernel / path | Result | % of 365 GB/s |
|---|---|---|
| qmv 8-bit, large (10240x2560, lm_head) | 234-236 GB/s | 64 % |
| qmv 8-bit, 2560x6144 | 283 GB/s | 77 % |
| qmv 8-bit, small (640x2560) | 129 GB/s | 35 % |
| gather_qmm, 10 of 512 experts (incl. top-k) | 88 GB/s | 24 % |
| Synthetic Flash-Next decode step, real shapes, mx.compile | 26.5 ms = 37.8 tok/s, 222 GB/s | 61 % |
| Real model: Qwen3.6-35B-A3B on mlx-lm, decode (1.67 GB/token) | 80-84 tok/s, ~136 GB/s | 37 % |
| Real model: Qwen3.6-35B-A3B prefill, 5,210-token prompt | 1,032 tok/s (~46 % of FP16 peak) | |

Flash-Next itself was not run: mlx-lm has no `qwen4_exp` loader, and installing the third-party
`mlx-serve` package was blocked by the session's permission policy.

## Gap to close (Flash-Next decode)
- Roofline 16.2 ms vs stock-op synthetic 26.5 ms; a real stock implementation would be slower still
  (35B-A3B reaches only 37 % of bandwidth end to end).
- Biggest single loss: the expert gather at 24 % of bandwidth -> fused MoE decode kernel.
- Small matvecs and ~1,500+ dispatches per token -> fuse per layer.
- Prefill estimate for Flash-Next (~13.3 GFLOP/token): ~470 tok/s at stock efficiency,
  ~700 tok/s at 70 % of FP16 peak; a 128K prompt takes ~3 min, so the prefix cache decides TTFT.

# Step 2a: fused MoE decode kernel (2026-11-02)

`bench/moe_kernel.py`: two Metal dispatches per layer (gate+up+SwiGLU, then down+weighted sum)
replace three `gather_qmm` calls plus SwiGLU and the weighted sum. FP32 accumulation and
activations; dense layers stay 8-bit (unchanged artifact).

Correctness against an FP32 oracle built from independently dequantised real layer-0 experts
(5 random routes): worst relative L2 error **5.1e-7** (stock MLX with bf16 activations: 4.7e-3).

Per-call kernel timing (10 of 512 experts, 48 calls per eval, distinct banks):
| Kernel | Bytes | Time | GB/s | Roofline (365 GB/s) |
|---|---|---|---|---|
| gate_up | 18.4 MB | 68 us | 270 | 50.5 us |
| down | 9.2 MB | 41 us | 223 | 25.2 us |

Tuning steps that mattered: half-simdgroup per row pair (+8 %), issuing loads before arithmetic,
and reusing each lane's activation chunk across 4 rows (activation reads in L1 had outweighed
weight reads 8:1; down 94 -> 41 us). Threadgroup size 4-8 simdgroups ties; 16 is slower.

Experts of 48 layers per token:
| Path | ms/token | GB/s |
|---|---|---|
| Roofline | 3.64 | 365 |
| Fused kernels alone | ~5.2 | ~255 |
| Fused, harness with extra elementwise ops | 6.15 | 216 |
| Stock gather_qmm route, same harness | 7.5 | 176 |

Projected decode with stock dense layers (~15.6-19 ms) + fused experts (5.2 ms): ~42-48 tok/s
(roofline 62). Dense layers (4.54 of 5.92 GB/token) are now the larger remaining gap.

# Step 2b: dense layers, 8-bit (2026-11-02)

`bench/dense_kernel.py`: `qmv8` (one token) and `qmm8` (T tokens per weight load), FP32
activations and accumulation; lanes per row (4-32) chosen per row length so K = 320 ... 10240 all
keep every lane busy, each lane's x chunk reused across 2-8 rows.

Correctness on real tensors (every row length the model uses, lm_head included), FP32 oracle:
worst relative L2 error **5.9e-7** (stock MLX bf16 path: 1.9e-3 to 2.8e-3). qmm8 T=5: 8.0e-7.

Per-shape, one token, best config (distinct copies per eval):
| Shape (N x K) | MB | stock | qmv8 |
|---|---|---|---|
| 16480 x 2560 (qkv+z+a+b grouped) | 44.8 | 246 GB/s | 295 GB/s |
| 13952 x 2560 (q+k+v+indexer grouped) | 37.9 | 303 | 295 |
| 2560 x 6144 (out / o proj) | 16.7 | 277 | 277 |
| 320 x 10240 (hyper-connection down) | 3.5 | 152 | 172 |
| 10240 x 320 (hyper-connection up) | 3.5 | 158 | 177 |
| 1792 x 2560 (shared gate+up+router) | 4.9 | 209 | 209 |
| 2560 x 640 (shared down) | 1.7 | 78 | 115 |
| 248320 x 2560 (lm_head) | 675 | 362 | 360 |

Stock MLX 8-bit matvec is already near the bandwidth roof on large matrices; the remaining loss is
small matrices (fixed ~10-15 us per kernel) and the dependency chain.

A token's whole dense set (4.533 GB, roofline 12.4 ms):
| Run | stock | qmv8 |
|---|---|---|
| Dependent chain, separate projections | 18.7 ms | 18.0 ms |
| Dependent chain, grouped projections | 20.1 ms | 19.0 ms |
| No dependencies (bandwidth only) | 15.7 ms (289 GB/s) | 15.7 ms |
Row-grouping did not help in the chain (sibling projections already overlap). The chain costs
~2.3-3 ms over pure bandwidth: ~400 dependent kernels per token.

Multi-token verification (us per call):
| Shape | T=1 stock/qmm8 | T=2 | T=4 | T=6 | T=8 |
|---|---|---|---|---|---|
| 16480 x 2560 | 184 / 149 | 212 / 157 | 276 / 220 | 426 / 315 | 425 / 422 |
| 2560 x 6144 | 62 / 62 | 76 / 73 | 93 / 94 | 174 / 123 | 175 / 162 |
| 320 x 10240 | 17 / 23 | 19 / 24 | 24 / 32 | 48 / 39 | 48 / 48 |
| 10240 x 320 | 24 / 23 | 38 / 21 | 70 / 28 | 40 / 33 | 40 / 39 |
| 2560 x 640 | 16 / 18 | 23 / 21 | 35 / 23 | 45 / 28 | 54 / 27 |
Two tokens cost about as much as one with qmm8; beyond 4 the kernel is limited by activation
reads and arithmetic, not weight bandwidth (M2 has no low-precision matrix units).

Updated single-stream projection: dense ~18 ms + experts ~5.2 ms + attention/GDN/elementwise
(not yet measured, est. 2-3 ms) = ~25 ms, ~40 tok/s without speculation (roofline 62).

# Step 3: full Qwen3.8-Flash-Next model (2026-11-02)

`engine/flash_next.py`: MLX implementation following transformers 5.22.1 `modular_qwen4_exp.py`
(hyper-connections x4, GatedDeltaNet via mlx-lm's kernel, gated attention with the QSA block
indexer, top-10 MoE + gated shared expert, PLE with the 32 GB n-gram table read by concurrent
pread from the external SSD). FP32 activations everywhere; artifact weights as stored.

Correctness:
- Passage perplexity 2.11 (119 tokens of English prose).
- Chinese chat answer correct and fluent (Rayleigh scattering, 3 sentences, stops at <|im_end|>).
- Needle retrieval correct at 1.2K, 3.6K, 11.6K and 24.7K tokens (sparse QSA selection active
  above 2,048 tokens).
- Passage perplexity 2.104 after the embedding fix below (was 2.11).
- Matches the transformers 5.22.1 reference (`engine/compare_hf.py`, `engine/compare_blocks.py`):
  the reference decoder layers run on CPU in FP32 with the artifact's weights dequantized exactly,
  on a 2,598-token chat prompt (past the 2,048-token indexer budget), with the engine prefilling
  in 512-token chunks and then decoding 8 tokens from its cache.
  | Check | Result |
  |---|---|
  | Per layer, fed the engine's input | 47 of 48 layers within 1e-5 relative (typically 4e-7 to 3e-6) |
  | The exception (layer 41) | one router near-tie: the two sides pick a different 10th expert for one token |
  | QSA indexer token selection | identical in all 12 attention layers (3.2M visible pairs each) |
  | N-gram hash constants, eos-aware shift | identical (shift: 6,000 random cases) |
  | Next-token argmax, all 2,598 positions | 99.92% agree (2 positions differ) |
  | Prompt perplexity | engine 2472.6, reference 2467.7 |
  | Decode logits vs reference | KL at most 6.6e-6, same argmax at all 9 steps |
  | Cached decode vs the engine's own prefill | 2.3e-6 relative |
  Block level (layer 0 and 42, real inputs): hyper-connections, linear attention, attention, router,
  shared and routed experts each agree to 1e-7 to 2e-6. Router ties are real: on layer 42, token 2341
  has the 10th and 11th expert probabilities exactly equal in FP32, so either choice is correct.
- Bug found by the comparison: the token embedding was dequantized with the stored bf16 scales,
  which makes `mx.dequantize` return bf16, so every embedding was rounded to bf16 (layer-0 error
  1.2e-3). Fixed by upcasting the scales; the dense and MoE kernels already did this.
- About the prompt perplexity: the model doesn't predict user-turn text (same file header: raw
  text 1.47, inside an assistant turn 1.38, inside a user turn 68), so the 2.5K figure reflects the
  prompt, not an error. Both sides agree on it.

Speed (one request, steady state, weights wired, Chrome etc. open):
| Context | Prefill | Decode |
|---|---|---|
| 512 | 375 tok/s | 27.5 tok/s (36.4 ms) |
| 4,096 | 328 tok/s | 26.2 tok/s |
| 12,288 | 316 tok/s | 25.8 tok/s |
Roofline: decode 62 tok/s (16.2 ms); prefill ~700 tok/s estimated.

Memory: 68.3-68.9 GiB active, peak 70.4-71.1 GiB (GPU working-set limit 77.8 GiB). The run had
5.2-5.6 GB of swap in use from other apps. The same benchmark before wiring the weights:
| Context | Prefill | Decode |
|---|---|---|
| 512 | 331 tok/s | 1.3 tok/s |
| 4,096 | 281 tok/s | 1.9 tok/s |
| 12,288 | 219 tok/s | 0.7 tok/s |
Wiring changes decode by 14-37x under memory pressure and prefill by 13-44%.

Fixes that mattered on the way (decode tok/s): file-backed weights re-read from the external
drive 0.6 -> copy into MLX memory 7.6 -> fused FP32 MoE kernel (stock gather_qmm is 4x slower
with FP32 inputs) + concurrent n-gram reads 22 -> mx.compile of elementwise blocks + async graph
build 25-28. Prefill: expert pairs sorted by expert (2.3x), logits for the last row only,
preallocated head-major KV. Long-context decode: gather the 2,048 QSA-selected tokens instead of
a T-wide mask (12K: 8 -> 26 tok/s). Memory: weights must be wired (mx.set_wired_limit); under
swap pressure decode fell to ~1 tok/s. MLX buffer cache capped at 2 GiB.

Decode step attribution (~36 ms, short context): MoE 13 (routed experts 5.1, router + shared
expert 7.9), linear attention 10.1, hyper-connections 7.8, full attention 3.9, PLE 1.5,
lm_head/sampling/residuals ~9.6.

## Qwen3.8-27B (4-bit) through the API server, vs other engines

M2 Max 96 GB, AC power, after a reboot (GPU 331 GB/s read, 7.3 TFLOPS bf16). Greedy, thinking off, 200 output tokens.

| Engine | Prefill (7.5K prompt) | Decode, short ctx | Decode, 7.5K ctx | Repeat prompt (edited last line) |
|---|---|---|---|---|
| engine/server.py, before | 92 tok/s | 12.8 | 13.5 | 39 s (4,096 reused) |
| engine/server.py, now (pipelined decode, 2048 prefill chunks) | 105 tok/s | 16.9 | 17.6 | 34 s (4,096 reused) |
| mlx-lm 0.32.2 `stream_generate` | 104 tok/s | 15.6 | 16.4 | no cache |
| LM Studio llama.cpp 2.67 (Q4_K_XL GGUF of a 27B fine-tune) | 95 tok/s | 8.5 (first call) | 12.6 | 5.6 s |

- Decode only falls from 18.6 to 16.5 tok/s between 1K and 24K context (in-process); sustained runs settle about 15% below the first second (clocks).
- The server's old 13 tok/s came from a synchronous loop (`.item()` every token). Sampling now runs on the GPU and the next forward is queued with `async_eval` first.
- llama.cpp reuses nearly the whole prompt when only the end changes; this server rolls back to the last 4,096-token snapshot (recurrent layers can't be truncated). In Claude Code's pattern (append-only) both reuse everything.
- Claude Code end to end on this model (before the pipelining fix): fixed a 2-bug test project in 8 min, 6 calls; first call 16.8K tokens prefill 163 s, later calls 2-15 s prefill thanks to the cache.

## big2flash 2.0 vs Mac engines (2026-11-03)

M2 Max 96 GB, Qwen3.8-27B 4-bit, greedy, no thinking. `tests/bench/run_all.sh` → `tests/bench/engines.jsonl`.
Best of 2 runs, each gated on a quiet GPU. Generation tok/s; long = 7.8K-token prompt, 200 out.

| Engine | Model file | Short | Long | Long prefill | Long TTFT |
|---|---|---|---|---|---|
| big2flash 2.0 (MTP k=1) | rapid-mlx 4-bit fp16, 15.1 GB | 27.37 | 28.15 | 178.6 | 43.7 s |
| big2flash (MTP off) | same | 22.15 | 20.88 | 175.1 | 44.6 s |
| Ollama 0.40.9 | qwen3.8:27b-mtp-q4_K_M | 19.28 | 21.21 | 138.1 | 56.6 s |
| mlx-lm 0.32.2 | same as big2flash | 19.22 | 18.16 | 172.1 | 45.4 s |
| LM Studio llama.cpp 2.68 (MTP) | huihui Q4_K_XL GGUF, ~17 GB | 12.39 | 14.10 | 127.0 | 61.7 s |
| LM Studio llama.cpp 2.68 | same | 12.33 | 13.33 | 122.9 | 63.9 s |

Plain-decode ceiling: 355 GB/s ÷ 15.1 GB ≈ 23.5 tok/s; MTP exceeds it. Greedy MTP output is token-identical to plain decoding.
Caveats: GGUF weights differ (size, abliterated fine-tune); MTP gain depends on content; ~±10% run-to-run noise.

### Re-run (2026-11-03 afternoon, installed app's server)

| Engine | Short | Long | Long prefill | Long TTFT |
|---|---|---|---|---|
| big2flash 2.0 (MTP k=1) | 24.04 | 25.33 | 179.1 | 43.6 s |
| big2flash (MTP off) | 22.15 | 21.07 | 179.5 | 43.5 s |
| Ollama 0.40.9 | 19.27 | 24.76 | 150.7 | 51.8 s |
| mlx-lm 0.32.2 | 19.41 | 18.27 | 187.0 | 41.7 s |
| LM Studio llama.cpp 2.68 | 14.77 | 15.66 | 142.2 | 55.2 s |
| LM Studio llama.cpp 2.68 (MTP) | 13.84 | 15.01 | 135.4 | 57.9 s |

MTP lead shrank vs the morning run (27.37/28.15); long-prompt gap to Ollama (+2%) is inside run-to-run noise.
