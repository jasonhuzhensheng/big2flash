"""8-bit quantized matvec for Qwen3.8-Flash-Next's dense layers on Apple GPUs (decode, one token).

y[n] = sum_k (scale[n, k/64] * q[n, k] + bias[n, k/64]) * x[k], MLX affine 8-bit, group 64,
4 values per uint32 low byte first. FP32 activations and accumulation.

LPR lanes share a row group of RB rows and stride across the row in 16-value chunks (one uint4);
each lane loads its x chunk once and reuses it for all RB rows. LPR is chosen per K so every lane
gets several chunks: the dense shapes run from K = 320 to K = 10240.
"""
import mlx.core as mx

GS = 64

_SRC = """
    constexpr uint CH = K / 16;          // uint4 chunks per row
    constexpr uint NC = CH / LPR;        // chunks per lane
    constexpr uint GPS = 32 / LPR;       // row groups per simdgroup
    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane % LPR;
    const uint grp = (threadgroup_position_in_grid.x * ROWS + simdgroup_index_in_threadgroup) * GPS + lane / LPR;
    const uint row0 = grp * RB;
    const device uint4* w4 = (const device uint4*)w;
    const device float4* x4 = (const device float4*)x;
    float acc[RB];
    for (uint rr = 0; rr < RB; rr++) acc[rr] = 0.0f;
    for (uint i = 0; i < NC; i++) {
        const uint c = sub + LPR * i;
        uint4 q[RB];
        float sc[RB], bi[RB];
        for (uint rr = 0; rr < RB; rr++) {
            const ulong row = metal::min(row0 + rr, uint(N) - 1u);
            q[rr] = w4[row * CH + c];
            sc[rr] = float(s[row * (K / 64) + (c >> 2)]);
            bi[rr] = float(b[row * (K / 64) + (c >> 2)]);
        }
        float4 xv[4];
        float xs = 0.0f;
        for (uint j = 0; j < 4; j++) {
            xv[j] = x4[c * 4 + j];
            xs += xv[j].x + xv[j].y + xv[j].z + xv[j].w;
        }
        for (uint rr = 0; rr < RB; rr++) {
            float d = 0.0f;
            for (uint j = 0; j < 4; j++) {
                d += metal::dot(xv[j], float4(as_type<uchar4>(q[rr][j])));
            }
            acc[rr] += sc[rr] * d + bi[rr] * xs;
        }
    }
    for (uint rr = 0; rr < RB; rr++) {
        float v = acc[rr];
        for (ushort off = LPR / 2; off > 0; off >>= 1) {
            v += simd_shuffle_down(v, off);
        }
        if (sub == 0 && row0 + rr < N) {
            y[row0 + rr] = v;
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="flash_next_qmv8", input_names=["x", "w", "s", "b"], output_names=["y"], source=_SRC)


def config(K):
    """(LPR, RB, ROWS) per row length; tuned in sweep_dense.py."""
    return CONFIGS.get(K, (8, 4, 2))


CONFIGS = {320: (4, 4, 4), 640: (8, 8, 2), 2560: (8, 4, 2), 6144: (16, 8, 2), 10240: (32, 4, 4)}


def qmv8(x, w, s, b, cfg=None):
    """x: (K,) float32; w: (N, K/4) uint32; s, b: (N, K/64). Returns (N,) float32."""
    N, K = w.shape[0], w.shape[1] * 4
    lpr, rb, rows = cfg or config(K)
    rows_per_tg = (32 // lpr) * rb * rows
    tgs = (N + rows_per_tg - 1) // rows_per_tg
    (y,) = _kernel(
        inputs=[x, w, s, b],
        template=[("N", N), ("K", K), ("LPR", lpr), ("RB", rb), ("ROWS", rows)],
        grid=(tgs * 32 * rows, 1, 1), threadgroup=(32 * rows, 1, 1),
        output_shapes=[(N,)], output_dtypes=[mx.float32])
    return y


def stock_qmv8(x, w, s, b):
    return mx.quantized_matmul(x.astype(mx.bfloat16)[None], w, s, b, transpose=True, group_size=GS, bits=8)[0]


def oracle_qmv8(x, w, s, b):
    W = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=GS, bits=8)
    return W @ x


# Multi-token (speculative verification) variant: T tokens share every weight load, so a round of
# T drafts reads the dense weights once. x is (T, K), y is (T, N).
_SRC_T = """
    constexpr uint CH = K / 16;
    constexpr uint NC = CH / LPR;
    constexpr uint GPS = 32 / LPR;
    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane % LPR;
    const uint grp = (threadgroup_position_in_grid.x * ROWS + simdgroup_index_in_threadgroup) * GPS + lane / LPR;
    const uint row0 = grp * RB;
    const device uint4* w4 = (const device uint4*)w;
    const device float4* x4 = (const device float4*)x;
    float acc[RB][T];
    for (uint rr = 0; rr < RB; rr++) for (uint t = 0; t < T; t++) acc[rr][t] = 0.0f;
    for (uint i = 0; i < NC; i++) {
        const uint c = sub + LPR * i;
        uint4 q[RB];
        float sc[RB], bi[RB];
        for (uint rr = 0; rr < RB; rr++) {
            const ulong row = metal::min(row0 + rr, uint(N) - 1u);
            q[rr] = w4[row * CH + c];
            sc[rr] = float(s[row * (K / 64) + (c >> 2)]);
            bi[rr] = float(b[row * (K / 64) + (c >> 2)]);
        }
        for (uint t = 0; t < T; t++) {
            float4 xv[4];
            float xs = 0.0f;
            for (uint j = 0; j < 4; j++) {
                xv[j] = x4[t * (K / 4) + c * 4 + j];
                xs += xv[j].x + xv[j].y + xv[j].z + xv[j].w;
            }
            for (uint rr = 0; rr < RB; rr++) {
                float d = 0.0f;
                for (uint j = 0; j < 4; j++) {
                    d += metal::dot(xv[j], float4(as_type<uchar4>(q[rr][j])));
                }
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
                y[t * N + row0 + rr] = v;
            }
        }
    }
"""

_kernel_t = mx.fast.metal_kernel(
    name="flash_next_qmm8_t", input_names=["x", "w", "s", "b"], output_names=["y"], source=_SRC_T)


def qmm8(x, w, s, b, cfg=None):
    """x: (T, K) float32 with small T (speculative verification); returns (T, N) float32."""
    T = x.shape[0]
    N, K = w.shape[0], w.shape[1] * 4
    lpr, rb, rows = cfg or config(K)
    rows_per_tg = (32 // lpr) * rb * rows
    tgs = (N + rows_per_tg - 1) // rows_per_tg
    (y,) = _kernel_t(
        inputs=[x, w, s, b],
        template=[("N", N), ("K", K), ("T", T), ("LPR", lpr), ("RB", rb), ("ROWS", rows)],
        grid=(tgs * 32 * rows, 1, 1), threadgroup=(32 * rows, 1, 1),
        output_shapes=[(T, N)], output_dtypes=[mx.float32])
    return y
