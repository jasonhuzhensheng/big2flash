"""4-bit quantized matvec / small-batch matmul for MLX affine 4-bit, group 64 (Qwen3.8-27B decode and
MTP verification).

y[t, n] = sum_k (scale[n, k/64] * q[n, k] + bias[n, k/64]) * x[t, k]; 8 values per uint32, low nibble
first. T rows of x (1 = decode, 2 = MTP verify) share every weight load, so verifying a draft reads the
weights once. FP32 accumulation, fp16 in/out.

LPR lanes share a group of RB rows and stride along K in 32-value chunks (one uint4 per row); each lane
loads its x chunk once per token and reuses it for all RB rows. Nibbles are taken with masks instead of
shifts: x is pre-scaled by 1, 1/16, 1/256, 1/4096 so q & 0x000f, 0x00f0, 0x0f00, 0xf000 of each 16-bit
half can be used unshifted.
"""
import mlx.core as mx
import mlx.nn as nn

_SRC = """
    constexpr uint CH = K / 32;          // uint4 chunks (32 values) per row
    constexpr uint NC = CH / LPR;        // chunks per lane
    constexpr uint GPS = 32 / LPR;       // row groups per simdgroup
    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane % LPR;
    const uint grp = (threadgroup_position_in_grid.x * ROWS + simdgroup_index_in_threadgroup) * GPS + lane / LPR;
    const uint row0 = grp * RB;
    const device uint4* w4 = (const device uint4*)w;
    float acc[RB][T];
    for (uint rr = 0; rr < RB; rr++) for (uint t = 0; t < T; t++) acc[rr][t] = 0.0f;
    for (uint i = 0; i < NC; i++) {
        const uint c = sub + LPR * i;
        uint4 q[RB];
        float sc[RB], bi[RB];
        for (uint rr = 0; rr < RB; rr++) {
            const ulong row = metal::min(row0 + rr, uint(N) - 1u);
            q[rr] = w4[row * CH + c];
            sc[rr] = float(s[row * (K / 64) + (c >> 1)]);
            bi[rr] = float(b[row * (K / 64) + (c >> 1)]);
        }
        // Unpack this chunk's weights once (WT: float or half), then stream the T input rows past them.
        vec<WT, 4> wv[RB][8];
        for (uint rr = 0; rr < RB; rr++) {
            for (uint j = 0; j < 4; j++) {
                const uint u = q[rr][j];
                wv[rr][2 * j] = vec<WT, 4>(u & 0xf, (u >> 4) & 0xf, (u >> 8) & 0xf, (u >> 12) & 0xf);
                wv[rr][2 * j + 1] = vec<WT, 4>((u >> 16) & 0xf, (u >> 20) & 0xf, (u >> 24) & 0xf, u >> 28);
            }
        }
        for (uint t = 0; t < T; t++) {
            const device half4* x4 = (const device half4*)(x + t * K + c * 32);
            const float xs = xsum[t * (K / 32) + c];  // sum of this chunk's 32 inputs, for the bias term
            vec<WT, 4> xv[8];
            for (uint j = 0; j < 8; j++) xv[j] = vec<WT, 4>(x4[j]);
            for (uint rr = 0; rr < RB; rr++) {
                float d = 0.0f;
                for (uint j = 0; j < 8; j += 2) d += float(metal::dot(xv[j], wv[rr][j]) + metal::dot(xv[j + 1], wv[rr][j + 1]));
                acc[rr][t] += sc[rr] * d + bi[rr] * xs;
            }
        }
    }
    for (uint rr = 0; rr < RB; rr++) {
        for (uint t = 0; t < T; t++) {
            float v = acc[rr][t];
            for (ushort off = LPR / 2; off > 0; off >>= 1) {
                v += simd_shuffle_down(v, off);
            }
            if (sub == 0 && row0 + rr < N) {
                y[t * N + row0 + rr] = half(v);
            }
        }
    }
"""

_kernel = mx.fast.metal_kernel(name="b2f_qmm4", input_names=["x", "w", "s", "b", "xsum"], output_names=["y"],
                               source=_SRC)

# (N, K, T) -> (LPR, RB, ROWS), from tests/bench/sweep_qmm4.py on M2 Max
CONFIGS = {
    (17408, 5120, 1): (8, 4, 2), (17408, 5120, 2): (4, 2, 2),
    (5120, 17408, 1): (8, 2, 2), (5120, 17408, 2): (8, 2, 2),
    (10240, 5120, 1): (8, 4, 2), (10240, 5120, 2): (4, 2, 2),
    (5120, 6144, 1): (8, 4, 1), (5120, 6144, 2): (4, 2, 2),
    (6144, 5120, 1): (8, 4, 4), (6144, 5120, 2): (4, 2, 2),
    (12288, 5120, 1): (8, 4, 2), (12288, 5120, 2): (4, 2, 2),
    (248320, 5120, 1): (8, 1, 1), (248320, 5120, 2): (8, 2, 1),
    (1024, 5120, 1): (8, 8, 2), (1024, 5120, 2): (4, 2, 1),
}


def config(N, K, T):
    if (N, K, T) in CONFIGS:
        return CONFIGS[(N, K, T)]
    return (8, 4, 2) if T == 1 else (4, 2, 2)


def qmm4(x, w, s, b, cfg=None):
    """x: (T, K) float16 with T small; w: (N, K/8) uint32; s, b: (N, K/64) float16. Returns (T, N) float16."""
    T, K = x.shape
    N = w.shape[0]
    lpr, rb, rows = cfg or config(N, K, T)
    rows_per_tg = (32 // lpr) * rb * rows
    tgs = (N + rows_per_tg - 1) // rows_per_tg
    (y,) = _kernel(
        inputs=[x, w, s, b, x.astype(mx.float32).reshape(T, K // 32, 32).sum(-1)],
        template=[("N", N), ("K", K), ("T", T), ("LPR", lpr), ("RB", rb), ("ROWS", rows), ("WT", WT)],
        grid=(tgs * 32 * rows, 1, 1), threadgroup=(32 * rows, 1, 1),
        output_shapes=[(T, N)], output_dtypes=[mx.float16])
    return y


WT = mx.float16  # unpacked weight / x type inside the kernel
MAX_T = 2  # beyond this the stock tiled kernel wins


def _ql_call(self, x):
    """nn.QuantizedLinear.__call__: decode-sized fp16 inputs on 4-bit/64 weights go to qmm4."""
    K = x.shape[-1]
    T = x.size // K
    if (T <= MAX_T and self.bits == 4 and self.group_size == 64 and x.dtype == mx.float16
            and "bias" not in self and getattr(self, "mode", "affine") == "affine" and K % 32 == 0 and (K // 32) % config(self.weight.shape[0], K, T)[0] == 0):
        y = qmm4(x.reshape(T, K), self.weight, self.scales, self.biases)
        return y.reshape(*x.shape[:-1], self.weight.shape[0])
    return _ql_orig(self, x)


_ql_orig = nn.QuantizedLinear.__call__


def install():
    nn.QuantizedLinear.__call__ = _ql_call
