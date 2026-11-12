"""4-bit (affine, group 64) matmul for 2..8 input rows on Apple GPUs: MTP verification of several drafts.

MLX's quantized matvec reads the weights once per input row, and from 6 rows on it switches to a tiled
kernel that runs at ~67 GB/s on M2 Max, so verifying k drafts costs nearly k times one decode step.
This kernel reads each weight once for up to 8 rows using the 8x8 simdgroup matrix units:

  C (8 out-rows x 8 inputs) += W (8 x 8 k) * Xt (8 k x 8 inputs)

Each lane unpacks the two weight values it owns in the W fragment straight from one uint4 load. The k order
inside a fragment is permuted so those two values sit in the lane's own 32-value chunk; x is permuted the
same way once per call (it is tiny), so the dot products are unchanged. Rows are padded to 8.
SPLIT simdgroups of a threadgroup take interleaved 128-k blocks of the same output rows (more parallelism
for the down projection, whose output is only 5120 wide) and are summed through threadgroup memory.
"""
import mlx.core as mx
import mlx.nn as nn

_SRC = """
    constexpr uint NB = K / 128;               // 128-k blocks per row
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint n0 = threadgroup_position_in_grid.x * (8 * R);
    const ushort qid = lane / 4;
    const ushort fm = (qid & 4) + ((lane / 2) % 4);     // this lane's row in an 8x8 fragment
    const ushort fn = (qid & 2) * 2 + (lane % 2) * 2;   // and its first column (it owns fn, fn+1)
    const device uint4* w4 = (const device uint4*)w;
    simdgroup_matrix<float, 8, 8> acc[R];
    for (uint r = 0; r < R; r++) acc[r] = simdgroup_matrix<float, 8, 8>(0.0f);
    // Next block's weights are loaded before this block's math, so the loads overlap the MMAs.
    uint4 qn[R];
    half scn[R], bin[R];
    for (uint r = 0; r < R; r++) {
        const uint row = n0 + 8 * r + fm;
        qn[r] = w4[row * (K / 32) + sg * 4 + fn / 2];
        scn[r] = s[row * (K / 64) + sg * 2 + fn / 4];
        bin[r] = b[row * (K / 64) + sg * 2 + fn / 4];
    }
    for (uint blk = sg; blk < NB; blk += SPLIT) {
        uint4 q[R];
        half sc[R], bi[R];
        const uint nxt = metal::min(blk + SPLIT, NB - 1);
        for (uint r = 0; r < R; r++) {
            q[r] = qn[r]; sc[r] = scn[r]; bi[r] = bin[r];
            const uint row = n0 + 8 * r + fm;
            qn[r] = w4[row * (K / 32) + nxt * 4 + fn / 2];
            scn[r] = s[row * (K / 64) + nxt * 2 + fn / 4];
            bin[r] = b[row * (K / 64) + nxt * 2 + fn / 4];
        }
        const device half* xb = xp + blk * (16 * 64);
        for (uint j = 0; j < 16; j++) {
            simdgroup_matrix<half, 8, 8> xm;
            simdgroup_load(xm, xb + j * 64, 8);
            const uint sh = 8 * (j % 4);
            for (uint r = 0; r < R; r++) {
                const uint u = q[r][j / 4] >> sh;
                simdgroup_matrix<half, 8, 8> wm;
                wm.thread_elements()[0] = sc[r] * half(u & 0xf) + bi[r];
                wm.thread_elements()[1] = sc[r] * half((u >> 4) & 0xf) + bi[r];
                simdgroup_multiply_accumulate(acc[r], wm, xm, acc[r]);
            }
        }
    }
    threadgroup float red[SPLIT * R * 64];
    for (uint r = 0; r < R; r++) simdgroup_store(acc[r], red + (sg * R + r) * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = sg * 32 + lane; e < R * 64; e += SPLIT * 32) {
        float v = 0.0f;
        for (uint p = 0; p < SPLIT; p++) v += red[p * R * 64 + e];
        const uint r = e / 64, n = (e % 64) / 8, t = e % 8;   // element (n, t) of row tile r
        y[t * N + n0 + 8 * r + n] = half(v);
    }
"""

_kernel = mx.fast.metal_kernel(name="b2f_qmm8", input_names=["xp", "w", "s", "b"], output_names=["y"],
                               source=_SRC)


def permute_x(x):
    """(T<=8, K) -> (K/128, 16, 8, 8): fragment j row c of block blk holds x[:, 128 blk + 32 (c//2) + 2 j + c%2]."""
    T, K = x.shape
    if T < 8:
        x = mx.concatenate([x, mx.zeros((8 - T, K), x.dtype)])
    return x.reshape(8, K // 128, 4, 16, 2).transpose(1, 3, 2, 4, 0).reshape(K // 128, 16, 8, 8)


def config(N, K):
    """(R row tiles per simdgroup, SPLIT simdgroups per threadgroup) — enough threadgroups to fill the GPU."""
    return (2, 2) if N >= 8192 else (2, 4)


def qmm8(x, w, s, b, cfg=None):
    """x: (T, K) float16, 2 <= T <= 8; w: (N, K/8) uint32; s, b: (N, K/64) float16. Returns (T, N) float16."""
    T, K = x.shape
    N = w.shape[0]
    R, split = cfg or config(N, K)
    (y,) = _kernel(
        inputs=[permute_x(x), w, s, b],
        template=[("N", N), ("K", K), ("R", R), ("SPLIT", split)],
        grid=(N // (8 * R) * 32 * split, 1, 1), threadgroup=(32 * split, 1, 1),
        output_shapes=[(8, N)], output_dtypes=[mx.float16])
    return y[:T]


def supported(layer, x):
    K = x.shape[-1]
    return (layer.bits == 4 and layer.group_size == 64 and x.dtype == mx.float16 and "bias" not in layer
            and getattr(layer, "mode", "affine") == "affine" and K % 128 == 0
            and layer.weight.shape[0] % 16 == 0)


MIN_T, MAX_T = 3, 8  # 1-2 rows: MLX's matvec is as fast


def _ql_call(self, x):
    K = x.shape[-1]
    T = x.size // K
    if MIN_T <= T <= MAX_T and supported(self, x):
        y = qmm8(x.reshape(T, K), self.weight, self.scales, self.biases)
        return y.reshape(*x.shape[:-1], self.weight.shape[0])
    return _ql_orig(self, x)


_ql_orig = nn.QuantizedLinear.__call__


def install():
    nn.QuantizedLinear.__call__ = _ql_call
