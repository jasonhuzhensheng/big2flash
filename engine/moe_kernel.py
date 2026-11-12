"""Fused MoE decode for Qwen3.8-Flash-Next on Apple GPUs (one token, top-K of E experts).

Two dispatches replace stock MLX's three gather_qmm calls plus SwiGLU and the weighted sum:
  gate_up: h[k, r] = silu(Wg[e_k, r] . x) * (Wu[e_k, r] . x)
  down:    y[d]    = sum_k g_k * (Wd[e_k, d] . h[k])
Weights are MLX affine 4-bit, group 64, packed 8 values per uint32 low bits first:
w = scale * q + bias. Accumulation is FP32.
"""
import mlx.core as mx

H, I, GS = 2560, 640, 64  # hidden, expert intermediate, group size
ROWS_PER_TG = 4  # simdgroups per threadgroup (sweep: 4 and 8 tie, 16 is slower)
RG, RD = 4, 4  # rows per half-simdgroup (gate_up), per simdgroup (down)

# Masking without shifting: (w & 0xF0) is 16 q, so x is pre-scaled by S = (1, 1/16, 1/256, 1/4096)
# (MLX qmv's trick). h is written pre-scaled the same way for the down projection.
_HEADER = """
constant float4 S = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
constant uint4 M = uint4(0xFu, 0xF0u, 0xF00u, 0xF000u);
inline float dot8(uint w, float4 a, float4 b) {  // a, b already scaled by S
    return metal::dot(a, float4(uint4(w & 0xFFFFu) & M)) + metal::dot(b, float4(uint4(w >> 16) & M));
}
"""

_GATE_UP_SRC = """
    // Half a simdgroup (16 lanes) per block of RG rows of one expert, gate and up together.
    // A lane loads its x chunk once and reuses it for all RG rows (x reads would otherwise
    // outweigh the weight reads in L1).
    const uint lane = thread_index_in_simdgroup;
    const uint l16 = lane & 15;
    const uint row0 = ((threadgroup_position_in_grid.x * ROWS + simdgroup_index_in_threadgroup) * 2 + (lane >> 4)) * RG;
    const uint k = row0 / I;
    const uint r0 = row0 % I;
    const ulong base_row = ulong(idx[k]) * I + r0;
    const device uint4* qg = (const device uint4*)wg;
    const device uint4* qu = (const device uint4*)wu;
    const device float4* x4 = (const device float4*)x;
    constexpr uint N = H / 32 / 16;  // x chunks (32 values, half a group) per lane
    float ag[RG], au[RG];
    for (uint rr = 0; rr < RG; rr++) { ag[rr] = 0.0f; au[rr] = 0.0f; }
    for (uint i = 0; i < N; i++) {
        const uint p = l16 + 16 * i;
        uint4 g4[RG], u4[RG];
        float gs[RG], gb[RG], us[RG], ub[RG];
        for (uint rr = 0; rr < RG; rr++) {
            const ulong row = base_row + rr;
            g4[rr] = qg[row * (H / 32) + p];
            u4[rr] = qu[row * (H / 32) + p];
            const ulong so = row * (H / 64) + (p >> 1);
            gs[rr] = float(sg[so]); gb[rr] = float(bg[so]);
            us[rr] = float(su[so]); ub[rr] = float(bu[so]);
        }
        float4 xsc[8];
        float xs = 0.0f;
        for (uint j = 0; j < 8; j++) {
            const float4 v = x4[p * 8 + j];
            xs += v.x + v.y + v.z + v.w;
            xsc[j] = v * S;
        }
        for (uint rr = 0; rr < RG; rr++) {
            float dg = 0.0f, du = 0.0f;
            for (uint c = 0; c < 4; c++) {
                dg += dot8(g4[rr][c], xsc[2 * c], xsc[2 * c + 1]);
                du += dot8(u4[rr][c], xsc[2 * c], xsc[2 * c + 1]);
            }
            ag[rr] += gs[rr] * dg + gb[rr] * xs;
            au[rr] += us[rr] * du + ub[rr] * xs;
        }
    }
    for (uint rr = 0; rr < RG; rr++) {
        float a = ag[rr], b = au[rr];
        for (ushort off = 8; off > 0; off >>= 1) {
            a += simd_shuffle_down(a, off);
            b += simd_shuffle_down(b, off);
        }
        if (l16 == 0) {
            h[row0 + rr] = a / (1.0f + metal::exp(-a)) * b * S[(r0 + rr) & 3];
        }
    }
"""

_DOWN_SRC = """
    // One simdgroup per RD consecutive output rows. Lanes take (expert, 32-value chunk) slots of
    // h, load each chunk once and reuse it for all RD rows. h arrives pre-scaled by S;
    // dot(h, 1/S) recovers the chunk's plain sum for the bias term.
    const uint lane = thread_index_in_simdgroup;
    const uint d0 = (threadgroup_position_in_grid.x * ROWS + simdgroup_index_in_threadgroup) * RD;
    const device float4* h4 = (const device float4*)h;
    const device uint4* wd4 = (const device uint4*)wd;
    constexpr uint C = I / 32;  // chunks per expert row
    const float4 U = float4(1.0f, 16.0f, 256.0f, 4096.0f);
    float acc[RD];
    for (uint rr = 0; rr < RD; rr++) acc[rr] = 0.0f;
    for (uint it = lane; it < K * C; it += 32) {
        const uint k = it / C;
        const uint w = it % C;
        const ulong row0 = ulong(idx[k]) * H + d0;
        uint4 q4[RD];
        float ds[RD], db[RD];
        for (uint rr = 0; rr < RD; rr++) {
            q4[rr] = wd4[(row0 + rr) * C + w];
            const ulong so = (row0 + rr) * (I / 64) + (w >> 1);
            ds[rr] = float(sd[so]);
            db[rr] = float(bd[so]);
        }
        float4 hv[8];
        float hs = 0.0f;
        for (uint j = 0; j < 8; j++) {
            hv[j] = h4[(k * I + w * 32) / 4 + j];
            hs += metal::dot(hv[j], U);
        }
        const float gk = gates[k];
        for (uint rr = 0; rr < RD; rr++) {
            float hq = 0.0f;
            for (uint c = 0; c < 4; c++) {
                hq += dot8(q4[rr][c], hv[2 * c], hv[2 * c + 1]);
            }
            acc[rr] += gk * (ds[rr] * hq + db[rr] * hs);
        }
    }
    for (uint rr = 0; rr < RD; rr++) {
        const float v = simd_sum(acc[rr]);
        if (lane == 0) {
            y[d0 + rr] = v;
        }
    }
"""

_gate_up = mx.fast.metal_kernel(
    name="flash_next_moe_gate_up",
    input_names=["x", "idx", "wg", "sg", "bg", "wu", "su", "bu"],
    output_names=["h"],
    source=_GATE_UP_SRC,
    header=_HEADER,
)
_down = mx.fast.metal_kernel(
    name="flash_next_moe_down",
    input_names=["h", "idx", "gates", "wd", "sd", "bd"],
    output_names=["y"],
    source=_DOWN_SRC,
    header=_HEADER,
)


def fused_moe(x, idx, gates, gate, up, down):
    """x: (H,) float32; idx: (K,) uint32; gates: (K,) float32; gate/up/down: (w, scales, biases)."""
    K = idx.shape[0]
    tmpl = [("H", H), ("I", I), ("K", K), ("ROWS", ROWS_PER_TG), ("RG", RG), ("RD", RD)]
    (h,) = _gate_up(
        inputs=[x, idx, *gate, *up], template=tmpl,
        grid=(K * I // (2 * RG) * 32, 1, 1), threadgroup=(32 * ROWS_PER_TG, 1, 1),
        output_shapes=[(K * I,)], output_dtypes=[mx.float32])
    (y,) = _down(
        inputs=[h, idx, gates, *down], template=tmpl,
        grid=(H // RD * 32, 1, 1), threadgroup=(32 * ROWS_PER_TG, 1, 1),
        output_shapes=[(H,)], output_dtypes=[mx.float32])
    return y


def stock_moe(x, idx, gates, gate, up, down):
    """mlx-lm SwitchGLU's route: three gather_qmm calls, SwiGLU, weighted sum."""
    xe = x.astype(mx.bfloat16).reshape(1, 1, 1, H)
    ix = idx.reshape(1, -1)
    g = mx.gather_qmm(xe, *gate, rhs_indices=ix, transpose=True, group_size=GS, bits=4)
    u = mx.gather_qmm(xe, *up, rhs_indices=ix, transpose=True, group_size=GS, bits=4)
    a = g * mx.sigmoid(g) * u
    y = mx.gather_qmm(a, *down, rhs_indices=ix, transpose=True, group_size=GS, bits=4)
    return (y.reshape(-1, H).astype(mx.float32) * gates[:, None]).sum(0)


def oracle_moe(x, idx, gates, gate, up, down):
    """FP32 reference from independently dequantised weights."""
    def deq(q):
        w, s, b = (a[idx] for a in q)
        return mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=GS, bits=4)
    wg, wu, wd = deq(gate), deq(up), deq(down)  # (K, I, H), (K, I, H), (K, H, I)
    g, u = wg @ x, wu @ x
    a = g * mx.sigmoid(g) * u
    return (gates[:, None] * (wd @ a[:, :, None])[..., 0]).sum(0)
