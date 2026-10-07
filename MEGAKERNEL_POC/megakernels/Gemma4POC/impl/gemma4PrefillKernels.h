#pragma once

#include <cstddef>

// Multi-token (prefill) path of the Gemma-4 MegaKernels. Appended to kFfnSource
// (it reuses REC_DW, rec_at, wg_sum, gelu_tanh, topk_softmax) and built as a
// separate program with the large register file.
//
// The decode kernels are GEMVs that re-read every weight once per token. Here
// all tokens of a call go through int4 XMX GEMMs on the same repacked records:
//  * One sub-group owns PF_MT tokens x 2 row blocks (32 rows). A record is
//    block-read once per 64-group and dequantised to f16 in registers, then fed
//    as the DPAS B operand to PF_MB 8-token A tiles.
//  * Dequantisation is a mask + or per dword: (w >> 4j) & 0x000F000F lands the
//    nibbles of k and k+4 in the two halves, and 0x6400 turns them into 1024+q.
//    Activations are therefore stored row-major with each 16-chunk permuted to
//    match ("P order": position l holds k = 8*(l/8) + (l%8)/2 + 4*(l%2)), and
//    the A tiles come straight from 2D block reads, which also zero the rows
//    past the end.
//  * The MoE runs as a grouped GEMM: (token, slot) pairs are bucketed per
//    expert into PF_MT-row tiles, gathered (plain row copies), and the down
//    projection scatters its weighted rows back per pair.
static const char* kPrefillBuildOpts = "-cl-intel-256-GRF-per-thread";

#ifndef PF_MB_DEFAULT
#define PF_MB_DEFAULT 8
#endif
// The program is built twice: kPfMb for the dense GEMMs and well-filled expert
// tiles, kPfMbSmall for short prompts and experts that get only a few rows.
constexpr int kPfMb = PF_MB_DEFAULT;    // 8-token blocks per sub-group
constexpr int kPfMt = 8 * kPfMb;        // tokens per GEMM tile
constexpr int kPfMbSmall = 1;
constexpr int kPfNsg = 8;               // sub-groups per GEMM work-group
constexpr size_t kPfLws = 16 * kPfNsg;
constexpr int kPfRouterKs = 11;         // router K slices (H / 256)

static const char* kPrefillSource = R"CLC(
#define PF_MT (8 * PF_MB)
#define PF_LWS (SG * PF_NSG)
#define PF_REQD __attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(PF_LWS, 1, 1)))

// Stores elements k = 8b..8b+7 of a P-order row.
inline void pf_store8(float8 v, int b, __global half* row) {
    vstore8(convert_half8((float8)(v.s0, v.s4, v.s1, v.s5, v.s2, v.s6, v.s3, v.s7)), b, row);
}

// DPAS B operand (16 k x 16 rows, one row per lane) from the two record
// dwords holding k 0..7 and 8..15 of this lane's row.
inline int8 pf_deq(uint lo, uint hi, half2 s, half2 z) {
    const uint M = 0x000F000Fu, B = 0x64006400u;
    int8 r;
    r.s0 = as_int((as_half2((lo & M) | B) - z) * s);
    r.s1 = as_int((as_half2(((lo >> 4) & M) | B) - z) * s);
    r.s2 = as_int((as_half2(((lo >> 8) & M) | B) - z) * s);
    r.s3 = as_int((as_half2(((lo >> 12) & M) | B) - z) * s);
    r.s4 = as_int((as_half2((hi & M) | B) - z) * s);
    r.s5 = as_int((as_half2(((hi >> 4) & M) | B) - z) * s);
    r.s6 = as_int((as_half2(((hi >> 8) & M) | B) - z) * s);
    r.s7 = as_int((as_half2(((hi >> 12) & M) | B) - z) * s);
    return r;
}

// The PF_MB A tiles (8 rows x 16 k each) of rows row0.. at k chunk kc of a
// P-order [rows, K] activation; rows >= `rows` read as zero.
#if PF_MB % 4 == 0
#define PF_LOAD_A(av, at, K, rows, row0, kc)                                                        \
    __attribute__((opencl_unroll_hint)) for (int q_ = 0; q_ < PF_MB / 4; ++q_)                     \
        intel_sub_group_2d_block_read_16b_32r16x1c((__global void*)(at), (K) * 2, (rows), (K) * 2,  \
                                                   (int2)((kc) * 16, (row0) + q_ * 32),             \
                                                   (__private ushort*)&av[q_ * 4]);
#else
#define PF_LOAD_A(av, at, K, rows, row0, kc)                                                        \
    __attribute__((opencl_unroll_hint)) for (int q_ = 0; q_ < PF_MB; ++q_)                         \
        intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)(at), (K) * 2, (rows), (K) * 2,   \
                                                  (int2)((kc) * 16, (row0) + q_ * 8),               \
                                                  (__private ushort*)&av[q_]);
#endif

// acc0/acc1[mb] += A(rows row0.., K) x records r0/r1 (two 16-row blocks).
#define PF_CORE(at, K, rows, row0, r0, r1, acc0, acc1)                                             \
    __attribute__((opencl_unroll_hint(1))) for (int g = 0; g < (K) / QG; ++g) {                   \
        const uint8 w0 = intel_sub_group_block_read8(r0 + g * REC_DW);                             \
        const uint m0 = intel_sub_group_block_read(r0 + g * REC_DW + 128);                         \
        const uint8 w1 = intel_sub_group_block_read8(r1 + g * REC_DW);                             \
        const uint m1 = intel_sub_group_block_read(r1 + g * REC_DW + 128);                         \
        const half2 s0 = (half2)(as_half2(m0).s0), z0 = (half2)((half)(1024 + (m0 >> 16)));        \
        const half2 s1 = (half2)(as_half2(m1).s0), z1 = (half2)((half)(1024 + (m1 >> 16)));        \
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < 4; ++c) {                          \
            ushort8 av_[PF_MB];                                                                    \
            PF_LOAD_A(av_, at, K, rows, row0, g * 4 + c)                                           \
            const int8 b0 = pf_deq(w0[2 * c], w0[2 * c + 1], s0, z0);                              \
            __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb)                 \
                acc0[mb] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(av_[mb]), b0, acc0[mb]); \
            const int8 b1 = pf_deq(w1[2 * c], w1[2 * c + 1], s1, z1);                              \
            __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb)                 \
                acc1[mb] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(av_[mb]), b1, acc1[mb]); \
        }                                                                                          \
    }

#define PF_ZERO(acc) __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb) acc[mb] = (float8)(0.0f);

// Column (lane) whose value sits at position l of a P-order 16-chunk.
inline int pf_pos_lane(int l) { return (l & 8) | ((l & 7) >> 1) | ((l & 1) << 2); }

// Writes acc (rows row0.., the sub-group's 16 columns = chunk c) as P-order f16.
inline void pf_store_rows(const float8 acc[PF_MB], int row0, int c, int K, int rows, __global half* out) {
    const int src = pf_pos_lane(get_sub_group_local_id());
    __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb) {
        ushort8 v;
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m)
            v[m] = as_ushort(convert_half(sub_group_shuffle(acc[mb][m], src)));
        intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)out, K * 2, rows, K * 2,
                                                   (int2)(c * 16, row0 + mb * 8), (__private ushort*)&v);
    }
}

// ---------------------------------------------------------------------------
// out[M, N] (f32, row stride ldo) = A[M, K] . W^T. Sub-group j: row blocks 2j, 2j+1.
// ---------------------------------------------------------------------------
PF_REQD __kernel void pf_gemm(const __global half* at, const int K, const __global uint* w, const int nrb,
                              const int M, __global float* out, const int ldo) {
    const int j = get_group_id(0) * PF_NSG + get_sub_group_id();
    if (2 * j >= nrb)
        return;
    const int row0 = get_group_id(1) * PF_MT;
    const int ln = get_sub_group_local_id();
    const __global uint* r0 = rec_at(w, K, 2 * j, 0);
    const __global uint* r1 = rec_at(w, K, 2 * j + 1, 0);
    float8 acc0[PF_MB], acc1[PF_MB];
    PF_ZERO(acc0) PF_ZERO(acc1)
    PF_CORE(at, K, M, row0, r0, r1, acc0, acc1)
    __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb) {
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const int row = row0 + mb * 8 + m;
            if (row < M) {
                out[(size_t)row * ldo + j * 32 + ln] = acc0[mb][m];
                out[(size_t)row * ldo + j * 32 + 16 + ln] = acc1[mb][m];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// act = gelu(gate . x) * (up . x), P-order f16 for the down projection. With
// tile_exp set (MoE), M tile t uses expert tile_exp[t] and only the first
// hdr[0] tiles exist; otherwise there are M rows.
// ---------------------------------------------------------------------------
PF_REQD __kernel void pf_gemm_gelu(const __global half* at, const int K, const __global uint* wg,
                                   const __global uint* wu, const int nrb, const int M, const __global int* hdr,
                                   const __global int* tile_exp, __global half* act) {
    const int rb = get_group_id(0) * PF_NSG + get_sub_group_id();
    const int mt = get_group_id(1);
    const int rows = hdr ? hdr[0] * PF_MT : M;
    if (rb >= nrb || mt * PF_MT >= rows)
        return;
    const int erb = (tile_exp ? tile_exp[mt] * nrb : 0) + rb;
    const __global uint* r0 = rec_at(wg, K, erb, 0);
    const __global uint* r1 = rec_at(wu, K, erb, 0);
    float8 acc0[PF_MB], acc1[PF_MB];
    PF_ZERO(acc0) PF_ZERO(acc1)
    PF_CORE(at, K, rows, mt * PF_MT, r0, r1, acc0, acc1)
    __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb) {
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m)
            acc0[mb][m] = gelu_tanh(acc0[mb][m]) * acc1[mb][m];
    }
    pf_store_rows(acc0, mt * PF_MT, rb, nrb * 16, rows, act);
}

// ---------------------------------------------------------------------------
// MoE down projection: ys[pair] = w[pair] * (down_e . act[row]) for every
// gathered row; padded rows (pair < 0) are dropped.
// ---------------------------------------------------------------------------
PF_REQD __kernel void pf_gemm_moe_down(const __global half* at, const int K, const __global uint* w,
                                       const int nrb, const __global int* hdr, const __global int* tile_exp,
                                       const __global int* pair_of_row, const __global float* tk_w,
                                       __global half* ys) {
    const int j = get_group_id(0) * PF_NSG + get_sub_group_id();
    const int mt = get_group_id(1);
    const int rows = hdr[0] * PF_MT;
    if (2 * j >= nrb || mt * PF_MT >= rows)
        return;
    const int ln = get_sub_group_local_id();
    const int erb = tile_exp[mt] * nrb + 2 * j;
    const __global uint* r0 = rec_at(w, K, erb, 0);
    const __global uint* r1 = rec_at(w, K, erb + 1, 0);
    float8 acc0[PF_MB], acc1[PF_MB];
    PF_ZERO(acc0) PF_ZERO(acc1)
    PF_CORE(at, K, rows, mt * PF_MT, r0, r1, acc0, acc1)
    const int N = nrb * 16;
    __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb) {
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const int p = pair_of_row[mt * PF_MT + mb * 8 + m];
            if (p >= 0) {
                const float tw = tk_w[p];
                ys[(size_t)p * N + j * 32 + ln] = convert_half(tw * acc0[mb][m]);
                ys[(size_t)p * N + j * 32 + 16 + ln] = convert_half(tw * acc1[mb][m]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Router logits, partial over K slice get_group_id(0):
//   out[ks][M][N] = A[:, slice] . W[:, slice]^T, dense f16 W [N, K], N = 16 * PF_NSG.
// ---------------------------------------------------------------------------
PF_REQD __kernel void pf_gemm_f16w(const __global half* at, const int K, const __global half* w, const int N,
                                   const int M, __global float* out) {
    const int rb = get_sub_group_id();
    const int ks = get_group_id(0);
    const int nks = get_num_groups(0);
    const int row0 = get_group_id(1) * PF_MT;
    const int ln = get_sub_group_local_id();
    const __global half* row = w + (size_t)(rb * 16 + ln) * K;
    const int c0 = ks * (K >> 4) / nks, c1 = (ks + 1) * (K >> 4) / nks;
    float8 acc[PF_MB];
    PF_ZERO(acc)
    __attribute__((opencl_unroll_hint(2))) for (int c = c0; c < c1; ++c) {
        const half16 h = vload16(c, row);
        const int8 b = (int8)(as_int((half2)(h.s0, h.s4)), as_int((half2)(h.s1, h.s5)),
                              as_int((half2)(h.s2, h.s6)), as_int((half2)(h.s3, h.s7)),
                              as_int((half2)(h.s8, h.sc)), as_int((half2)(h.s9, h.sd)),
                              as_int((half2)(h.sa, h.se)), as_int((half2)(h.sb, h.sf)));
        ushort8 av[PF_MB];
        PF_LOAD_A(av, at, K, M, row0, c)
        __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb)
            acc[mb] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(av[mb]), b, acc[mb]);
    }
    __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb)
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const int r = row0 + mb * 8 + m;
            if (r < M)
                out[((size_t)ks * M + r) * N + rb * 16 + ln] = acc[mb][m];
        }
}

// ---------------------------------------------------------------------------
// Natural-order [M, K] -> P-order, one work-item per 16-chunk.
// ---------------------------------------------------------------------------
__kernel void pf_tile(const __global half* src, const int K, const int M, __global half* dst) {
    const int c = get_global_id(0);
    const int r = get_global_id(1);
    if (c >= K / 16 || r >= M)
        return;
    const ushort16 x = vload16(c, (const __global ushort*)src + (size_t)r * K);
    vstore16((ushort16)(x.s0, x.s4, x.s1, x.s5, x.s2, x.s6, x.s3, x.s7,
                        x.s8, x.sc, x.s9, x.sd, x.sa, x.se, x.sb, x.sf), c, (__global ushort*)dst + (size_t)r * K);
}

// ---------------------------------------------------------------------------
// MoE gather: row r = src row pair_of_row[r] / TOPK_ (zeros for padding) of a
// [*, K] f16 activation; only hdr[0] tiles exist. Sub-group i of row r copies
// 512-byte block i.
// ---------------------------------------------------------------------------
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gather(const __global uint* src, const int K, const __global int* hdr,
                        const __global int* pair_of_row, __global uint* dst) {
    const int r = get_group_id(1);
    if (r >= hdr[0] * PF_MT)
        return;
    const int p = pair_of_row[r];
    const size_t words = K / 2;
    const int i = get_sub_group_id() * 128;
    const uint8 v = p < 0 ? (uint8)(0) : intel_sub_group_block_read8(src + (size_t)(p / TOPK_) * words + i);
    intel_sub_group_block_write8(dst + r * words + i, v);
}

// ---------------------------------------------------------------------------
// One work-group per token: RMS(x) * w_in, P-order (the q|k|v GEMM input).
// ---------------------------------------------------------------------------
REQD __kernel void pf_attn_rms(const __global half* x, const __global half* w_in, __global half* at) {
    __local float tmp[NSG];
    const int t = get_group_id(0);
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    x += (size_t)t * H_;
    float8 hv[HB_PER_LANE];
    float acc2 = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        hv[i] = b < HB ? vload_half8(b, x) : (float8)(0.0f);
        acc2 += dot(hv[i].lo, hv[i].lo) + dot(hv[i].hi, hv[i].hi);
    }
    const float inv = rsqrt(wg_sum(acc2, tmp) / H_ + EPS_);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        if (b < HB)
            pf_store8(hv[i] * inv * vload_half8(b, w_in), b, at + (size_t)t * H_);
    }
}

// ---------------------------------------------------------------------------
// One work-group per token, after o_proj:
//   h1 = h + RMS(o) * w_pa     (f32)
//   xd = RMS(h1) * w_pf        dense MLP input
//   xr = RMS(h1) * w_r         router input
//   x2 = RMS(h1) * w_pf2       MoE input, gathered per expert later
// (all three P-order f16)
// ---------------------------------------------------------------------------
REQD __kernel void pf_ffn_in(const __global float* o, const __global half* h,
                             const __global half* w_pa, const __global half* w_pf,
                             const __global half* w_r, const __global half* w_pf2,
                             __global float* h1_out, __global half* xd, __global half* xr,
                             __global half* x2) {
    __local float tmp[NSG];
    const int t = get_group_id(0);
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    o += (size_t)t * H_;
    h += (size_t)t * H_;
    float8 hv[HB_PER_LANE];
    float acc2 = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        hv[i] = b < HB ? vload8(b, o) : (float8)(0.0f);
        acc2 += dot(hv[i].lo, hv[i].lo) + dot(hv[i].hi, hv[i].hi);
    }
    const float inv_o = rsqrt(wg_sum(acc2, tmp) / H_ + EPS_);
    acc2 = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        if (b < HB)
            hv[i] = vload_half8(b, h) + hv[i] * inv_o * vload_half8(b, w_pa);
        acc2 += dot(hv[i].lo, hv[i].lo) + dot(hv[i].hi, hv[i].hi);
    }
    const float inv1 = rsqrt(wg_sum(acc2, tmp) / H_ + EPS_);
    const size_t ro = (size_t)t * H_;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        if (b < HB) {
            vstore8(hv[i], b, h1_out + ro);
            const float8 n = hv[i] * inv1;
            pf_store8(n * vload_half8(b, w_pf), b, xd + ro);
            pf_store8(n * vload_half8(b, w_r), b, xr + ro);
            pf_store8(n * vload_half8(b, w_pf2), b, x2 + ro);
        }
    }
}

// One sub-group per token: sums the KS router partials (into slice 0; every
// lane reads back only what it wrote), then the top-k softmax.
REQD __kernel void pf_topk(__global float* logits, const int T, const int KS, __global int* tk_idx,
                           __global float* tk_w) {
    const int t = get_group_id(0) * NSG + get_sub_group_id();
    if (t >= T)
        return;
    __global float* l = logits + (size_t)t * E_;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < E_ / SG; ++i) {
        const int n = get_sub_group_local_id() + SG * i;
        float v = l[n];
        for (int ks = 1; ks < KS; ++ks)
            v += l[(size_t)ks * T * E_ + n];
        l[n] = v;
    }
    topk_softmax(l, 0, true, tk_idx + t * TOPK_, tk_w + t * TOPK_);
}

// ---------------------------------------------------------------------------
// Buckets the T*TOPK_ (token, slot) pairs per expert into PF_MT-row tiles.
// hdr[0] = tile count; tile_exp[tile] = expert; pair_of_row[row] = pair or -1.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void pf_moe_route(const __global int* tk_idx, const int T, __global int* hdr, __global int* tile_exp,
                           __global int* pair_of_row) {
    __local int cnt[E_], base[E_], fill[E_];
    const int lid = get_local_id(0);
    const int P = T * TOPK_;
    for (int e = lid; e < E_; e += 1024) {
        cnt[e] = 0;
        fill[e] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = lid; p < P; p += 1024)
        atomic_inc(&cnt[tk_idx[p]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int acc = 0;
        for (int e = 0; e < E_; ++e) {
            base[e] = acc;
            acc += (cnt[e] + PF_MT - 1) / PF_MT;
        }
        hdr[0] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int e = lid; e < E_; e += 1024) {
        const int nt = (cnt[e] + PF_MT - 1) / PF_MT;
        for (int i = 0; i < nt; ++i)
            tile_exp[base[e] + i] = e;
        for (int r = cnt[e]; r < nt * PF_MT; ++r)
            pair_of_row[base[e] * PF_MT + r] = -1;
    }
    for (int p = lid; p < P; p += 1024) {
        const int e = tk_idx[p];
        pair_of_row[base[e] * PF_MT + atomic_inc(&fill[e])] = p;
    }
}

// ---------------------------------------------------------------------------
// One work-group per token: the ffn_out tail with e = sum of the weighted
// expert rows ys[t * TOPK_ + slot].
// ---------------------------------------------------------------------------
REQD __kernel void pf_ffn_out(const __global float* h1, const __global float* m, const __global half* ys,
                              const __global half* w1, const __global half* w2, const __global half* w_post,
                              const __global half* layer_scalar, __global half* out) {
    __local float tmp[NSG];
    const int t = get_group_id(0);
    const int lid = get_local_id(0);
    h1 += (size_t)t * H_;
    m += (size_t)t * H_;
    ys += (size_t)t * TOPK_ * H_;
    out += (size_t)t * H_;
    float mv[HPT], ev[HPT];
    float sm = 0.0f, se = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT; ++i) {
        const int k = lid + i * LWS;
        mv[i] = m[k];
        float e = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int s = 0; s < TOPK_; ++s)
            e += vload_half(s * H_ + k, ys);
        ev[i] = e;
        sm += mv[i] * mv[i];
        se += ev[i] * ev[i];
    }
    const float inv_m = rsqrt(wg_sum(sm, tmp) / H_ + EPS_);
    const float inv_e = rsqrt(wg_sum(se, tmp) / H_ + EPS_);
    float ss = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT; ++i) {
        const int k = lid + i * LWS;
        mv[i] = mv[i] * inv_m * vload_half(k, w1) + ev[i] * inv_e * vload_half(k, w2);
        ss += mv[i] * mv[i];
    }
    const float inv_s = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
    const float ls = vload_half(0, layer_scalar);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT; ++i) {
        const int k = lid + i * LWS;
        vstore_half((h1[k] + mv[i] * inv_s * vload_half(k, w_post)) * ls, k, out);
    }
}
)CLC";
