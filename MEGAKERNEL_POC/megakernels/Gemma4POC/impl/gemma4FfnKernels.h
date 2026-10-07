#pragma once

#include <cstddef>

// Shape -D flags are added by the caller (H_, I_, IE_, E_, TOPK_, EPS_).
static const char* kFfnBuildOpts = "-cl-std=CL3.0 -cl-mad-enable";
static constexpr size_t kFfnLws = 256;

// Work-groups of ffn_oproj, ffn_in, ffn_mid, ffn_moe_down, ffn_out; must match
// the *_WGS macros of the source below.
constexpr int kFfnKernels = 5;
inline void ffn_global_sizes(int H, int I, int IE, int E, int topk, size_t gws[kFfnKernels]) {
    gws[0] = kFfnLws * (H / 32);                     // OPROJ_WGS
    gws[1] = kFfnLws * (I / 64 + E / 16);            // DENSE_WGS + ROUTER_WGS
    gws[2] = kFfnLws * (topk * (IE / 64) + H / 32);  // MOE_WGS + DOWN_WGS
    gws[3] = kFfnLws * (H / 32);                     // EDOWN_WGS
    gws[4] = kFfnLws;
}

// OpenCL source of the Gemma-4 FFN-block MegaKernel (see gemma4POCRuntime.h).
// Built with -DH_ -DI_ -DIE_ -DE_ -DTOPK_ -DEPS_ plus kFfnBuildOpts.
//
// int4 GEMV design, each step measured on B60 with research/gemma4_ffn_bench:
//  * v1, 16 lanes per row with f32 dequant: ALU bound at 100-150 GB/s (u8->f32
//    is 3 movs per element on Xe2, scale shuffles used indirect addressing).
//  * v2, one row per lane: the activation becomes uniform across the
//    sub-group and is consumed as a broadcast operand. It is pre-quantised into
//    two int8 planes per 64-element group, x ~= a * (xh + xl / 254), which keeps
//    ~16 bits (more than the f16 the stock graph carries) and turns the inner
//    loop into packed dp4a, 7 integer ops per 8 weights. Zero points use
//    sum((q - z) x) = sum(q x) - z sum(x) with sum(x) over the quantised x, so
//    the identity is exact. Still only ~200 GB/s: each lane gathered its row's
//    32 bytes, scale and zero point from 16 different rows.
//  * v3 (this): weights are repacked at compile time (gemma4Repack.h) so that
//    a 16-row block's group is one contiguous 576-byte record - weights plus
//    packed scale|zero point - read with two sub-group block reads.
//  * Row counts too small to fill the GPU are split along K across the
//    sub-groups of a work-group and reduced through SLM.
static const char* kFfnSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define SG 16
#define NSG 16
#define LWS (SG * NSG)
#define QG 64
#define XL_RES 254.0f
#define REC_DW 144                            // repacked record: 128 weight + 16 meta dwords

#define REQD __attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(LWS, 1, 1)))

#ifndef GEMV_UNROLL
#define GEMV_UNROLL 4
#endif

// Work-group split; ffn_global_sizes() on the host must match.
#define DENSE_KS 4                            // ffn_in dense: 4 row blocks x 4 K slices
#define DENSE_WGS (I_ / (SG * NSG / DENSE_KS))
#define ROUTER_KS 16                          // ffn_in router: 1 row block x 16 K slices
#define ROUTER_WGS (E_ / SG)
#define MOE_KS 4                              // ffn_mid experts: 4 row blocks x 4 K slices
#define MOE_BLK (IE_ / (SG * NSG / MOE_KS))
#define MOE_WGS (TOPK_ * MOE_BLK)
#define DOWN_KS 8                             // ffn_mid dense down: 2 row blocks x 8 K slices
#define DOWN_WGS (H_ / (SG * NSG / DOWN_KS))
#define EDOWN_RB 2                            // ffn_moe_down: 2 row blocks x TOPK_ slots
#define EDOWN_WGS (H_ / (SG * EDOWN_RB))

// Quantised activation layout (K elements): xh[K/4] | xl[K/4] | xa[K/64] | xs[K/64].
// xh/xl dword 2b = even elements of 8-block b, 2b+1 = odd elements; xa[g] is the
// dequant scale and xs[g] the sum of the dequantised group g.
#define XQ_WORDS(K) ((K) / 2 + 2 * ((K) / QG))

inline float wg_sum(float v, __local float* tmp) {
    v = sub_group_reduce_add(v);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_local_id() == 0)
        tmp[get_sub_group_id()] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    return sub_group_reduce_add(tmp[get_sub_group_local_id()]);
}

inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

inline uint pack4(int a, int b, int c, int d) {
    return (uint)(a & 255) | ((uint)(b & 255) << 8) | ((uint)(c & 255) << 16) | ((uint)(d & 255) << 24);
}

typedef struct {
    uint he, ho, le, lo;
    float a, gsum;
} QB;

// Quantises the 8 consecutive elements of one 8-block. The 16 lanes of the
// sub-group must hold 16 consecutive blocks starting at a multiple of 16, so
// each 8-lane cluster is one 64-element group; invalid lanes still take part
// in the cluster reductions.
inline QB quant_block(float8 v, bool valid) {
    const float8 av = fabs(v);
    float m = fmax(fmax(fmax(av.s0, av.s1), fmax(av.s2, av.s3)), fmax(fmax(av.s4, av.s5), fmax(av.s6, av.s7)));
    m = sub_group_clustered_reduce_max(valid ? m : 0.0f, 8);
    QB q;
    q.a = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
    const float8 t = v * (1.0f / q.a);
    const float8 hf = rint(t);
    const int8 hi = convert_int8(hf);
    const int8 lo = convert_int8(rint((t - hf) * XL_RES));
    const int sh = sub_group_clustered_reduce_add(valid ? hi.s0 + hi.s1 + hi.s2 + hi.s3 + hi.s4 + hi.s5 + hi.s6 + hi.s7 : 0, 8);
    const int sl = sub_group_clustered_reduce_add(valid ? lo.s0 + lo.s1 + lo.s2 + lo.s3 + lo.s4 + lo.s5 + lo.s6 + lo.s7 : 0, 8);
    q.gsum = q.a * ((float)sh + (float)sl * (1.0f / XL_RES));
    q.he = pack4(hi.s0, hi.s2, hi.s4, hi.s6);
    q.ho = pack4(hi.s1, hi.s3, hi.s5, hi.s7);
    q.le = pack4(lo.s0, lo.s2, lo.s4, lo.s6);
    q.lo = pack4(lo.s1, lo.s3, lo.s5, lo.s7);
    return q;
}

#define DEFINE_STORE_QB(NAME, AS)                                                   \
inline void NAME(QB q, int b, int K, AS uint* x) {                                  \
    x[2 * b] = q.he;                                                                \
    x[2 * b + 1] = q.ho;                                                            \
    x[K / 4 + 2 * b] = q.le;                                                        \
    x[K / 4 + 2 * b + 1] = q.lo;                                                    \
    if ((get_sub_group_local_id() & 7) == 0) {                                      \
        x[K / 2 + (b >> 3)] = as_uint(q.a);                                         \
        x[K / 2 + K / QG + (b >> 3)] = as_uint(q.gsum);                             \
    }                                                                               \
}
DEFINE_STORE_QB(store_qb_l, __local)
DEFINE_STORE_QB(store_qb_g, __global)

// Quantises x[0..K) (global, loaded with LOAD8(b) as float8) into SLM;
// executed by the whole work-group.
#define QUANT_GLOBAL_WITH(LOAD8, K, xq)                                             \
    for (int c0_ = get_sub_group_id() * SG; c0_ < (K) / 8; c0_ += NSG * SG) {       \
        const int b_ = c0_ + get_sub_group_local_id();                             \
        const bool ok_ = b_ < (K) / 8;                                             \
        const QB q_ = quant_block(ok_ ? LOAD8(b_) : (float8)(0.0f), ok_);          \
        if (ok_)                                                                   \
            store_qb_l(q_, b_, (K), (xq));                                         \
    }

// One group of 64 weights (8 dwords of this lane's row) against the group's
// broadcast activation dwords.
#ifdef DOT_TRIVIAL  // harness-only: keeps every load, drops the math
#define DOT8(wv, xhv, xlv, ih, il) \
    ih += (int)((wv).s0 ^ (wv).s1 ^ (wv).s2 ^ (wv).s3 ^ (wv).s4 ^ (wv).s5 ^ (wv).s6 ^ (wv).s7 ^ (xhv) ^ (xlv));
#else
#define DOT8(wv, xhv, xlv, ih, il)                                                          \
    __attribute__((opencl_unroll_hint)) for (int j = 0; j < 8; ++j) {                       \
        const uint lo = (wv)[j] & 0x0F0F0F0Fu;                                              \
        const uint hi = ((wv)[j] >> 4) & 0x0F0F0F0Fu;                                       \
        ih += dot_4x8packed_us_int(lo, sub_group_broadcast(xhv, 2 * j)) +                   \
              dot_4x8packed_us_int(hi, sub_group_broadcast(xhv, 2 * j + 1));                \
        il += dot_4x8packed_us_int(lo, sub_group_broadcast(xlv, 2 * j)) +                   \
              dot_4x8packed_us_int(hi, sub_group_broadcast(xlv, 2 * j + 1));                \
    }
#endif

// NR row blocks per sub-group (lane l = row l of each block, sharing the
// activation) over ng groups. rec[r] is the block's record of its first group
// (group g0 of the activation xq of K elements, in address space AS).
#define DEFINE_ROWDOT(NAME, NR, AS)                                                         \
inline void NAME(const __global uint* rec[NR], const int g0, const int ng,                  \
                 const AS uint* xq, const int K, float acc[NR]) {                           \
    const AS uint* xh = xq;                                                                 \
    const AS uint* xl = xq + K / 4;                                                         \
    const AS float* xa = (const AS float*)(xq + K / 2);                                     \
    const AS float* xs = xa + K / QG;                                                       \
    __attribute__((opencl_unroll_hint)) for (int r = 0; r < NR; ++r) acc[r] = 0.0f;        \
    __attribute__((opencl_unroll_hint(GEMV_UNROLL))) for (int gi = 0; gi < ng; ++gi) {      \
        const int g = g0 + gi;                                                              \
        const uint xhv = intel_sub_group_block_read(xh + g * 16);                           \
        const uint xlv = intel_sub_group_block_read(xl + g * 16);                           \
        const float a = xa[g];                                                              \
        const float sx = xs[g];                                                             \
        __attribute__((opencl_unroll_hint)) for (int r = 0; r < NR; ++r) {                  \
            const __global uint* p = rec[r] + gi * REC_DW;                                  \
            const uint8 wv = intel_sub_group_block_read8(p);                                \
            const uint meta = intel_sub_group_block_read(p + 128);                          \
            int ih = 0, il = 0;                                                             \
            DOT8(wv, xhv, xlv, ih, il)                                                      \
            const float sc = (float)as_half2(meta).s0;                                      \
            const float zz = (float)(meta >> 16);                                           \
            acc[r] = fma(sc, fma(a, fma((float)il, 1.0f / XL_RES, (float)ih), -zz * sx), acc[r]); \
        }                                                                                   \
    }                                                                                       \
}

DEFINE_ROWDOT(rowdot1, 1, __local)
DEFINE_ROWDOT(rowdot2, 2, __local)
DEFINE_ROWDOT(rowdot1g, 1, __global)
DEFINE_ROWDOT(rowdot2g, 2, __global)

// Weight records loaded ahead of a kernel's activation prologue, so their DRAM
// latency overlaps the RMS/quantisation work instead of following it.
#ifndef PRE_G
#define PRE_G 4
#endif
typedef struct {
    uint8 w[PRE_G];
    uint m[PRE_G];
} PreRec;

inline PreRec preload(const __global uint* rec, int ng) {
    PreRec p;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < PRE_G; ++i) {
        if (i < ng) {
            p.w[i] = intel_sub_group_block_read8(rec + i * REC_DW);
            p.m[i] = intel_sub_group_block_read(rec + i * REC_DW + 128);
        }
    }
    return p;
}

// acc += the first min(PRE_G, ng) groups from `p`; the caller continues with
// rowdot over the remaining groups.
inline float predot(const PreRec* p, const int g0, const int ng, const __local uint* xq, const int K) {
    const __local uint* xh = xq;
    const __local uint* xl = xq + K / 4;
    const __local float* xa = (const __local float*)(xq + K / 2);
    const __local float* xs = xa + K / QG;
    float acc = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int gi = 0; gi < PRE_G; ++gi) {
        if (gi < ng) {
            const int g = g0 + gi;
            const uint xhv = intel_sub_group_block_read(xh + g * 16);
            const uint xlv = intel_sub_group_block_read(xl + g * 16);
            const uint8 wv = p->w[gi];
            int ih = 0, il = 0;
            DOT8(wv, xhv, xlv, ih, il)
            const float sc = (float)as_half2(p->m[gi]).s0;
            const float zz = (float)(p->m[gi] >> 16);
            acc = fma(sc, fma(xa[g], fma((float)il, 1.0f / XL_RES, (float)ih), -zz * xs[g]), acc);
        }
    }
    return acc;
}

// Record of row block rb, group g in a repacked [rows, K] matrix.
inline const __global uint* rec_at(const __global uint* m, int K, int rb, int g) {
    return m + ((size_t)rb * (K / QG) + g) * REC_DW;
}

// A work-group that produced the 64 outputs of quantisation group g (in SLM)
// quantises them straight into the next kernel's activation buffer, so that
// kernel needs no SLM prologue or barrier. Called by one full sub-group.
inline void quant_group_store(const __local float* vals, int g, int K, __global uint* xq) {
    const int ln = get_sub_group_local_id();
    const bool ok = ln < 8;
    const QB q = quant_block(ok ? vload8(ln, vals) : (float8)(0.0f), ok);
    if (ok)
        store_qb_g(q, g * 8 + ln, K, xq);
}

// Softmax over the top-k router logits, computed by one sub-group; ties go to
// the lower expert index, as in the stock MoERouterFused kernel. Returns the
// expert of slot `want`; with store set, writes all slots out.
inline int topk_softmax(const __global float* logits, int want, bool store, __global int* tk_idx,
                        __global float* tk_w) {
    const int ln = get_sub_group_local_id();
    float v[E_ / SG];
    // NaN logits must not produce an out-of-range expert index.
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < E_ / SG; ++i) {
        const float l = logits[ln + SG * i];
        v[i] = isnan(l) ? -INFINITY : l;
    }
    float top = 0.0f, wsum = 0.0f, mine = 0.0f;
    int mine_idx = 0, wanted = 0;
    __attribute__((opencl_unroll_hint)) for (int t = 0; t < TOPK_; ++t) {
        float lm = v[0];
        int li = ln;
        __attribute__((opencl_unroll_hint)) for (int i = 1; i < E_ / SG; ++i) {
            if (v[i] > lm) {
                lm = v[i];
                li = ln + SG * i;
            }
        }
        const float mx = sub_group_reduce_max(lm);
        const int idx = sub_group_reduce_min(lm == mx ? li : 0x7fffffff);
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < E_ / SG; ++i)
            if (ln + SG * i == idx)
                v[i] = -INFINITY;
        if (t == 0)
            top = mx;
        const float e = exp(mx - top);
        wsum += e;
        if (ln == t) {
            mine = e;
            mine_idx = idx;
        }
        if (t == want)
            wanted = idx;
    }
    if (store && ln < TOPK_) {
        tk_idx[ln] = mine_idx;
        tk_w[ln] = mine / wsum;
    }
    return wanted;
}

// ---------------------------------------------------------------------------
// ffn_oproj: o = o_proj(attn), the attention output projection ([H, KA], KA =
// heads * head_dim differs between sliding and full-attention layers).
// ---------------------------------------------------------------------------
#define OPROJ_KMAX 8192
#define OPROJ_KS 8                            // 2 row blocks x 8 K slices
#define OPROJ_WGS (H_ / (SG * NSG / OPROJ_KS))

REQD __kernel void ffn_oproj(const __global half* attn, const __global uint* opk, const int KA,
                             __global float* o_out) {
    __local uint xq[XQ_WORDS(OPROJ_KMAX)];
    __local float red[NSG][SG];
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const int ks = sg % OPROJ_KS;
    const int rb = get_group_id(0) * (NSG / OPROJ_KS) + sg / OPROJ_KS;
    const int G = KA / QG;
    const int g0 = ks * G / OPROJ_KS;
    const int g1 = (ks + 1) * G / OPROJ_KS;
    const __global uint* rec0 = rec_at(opk, KA, rb, g0);
#if PRE_G
    const PreRec pre = preload(rec0, g1 - g0);
#endif
#define LOAD_ATTN(b) vload_half8((b), attn)
    QUANT_GLOBAL_WITH(LOAD_ATTN, KA, xq)
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[1];
#if PRE_G
    const __global uint* rec[1] = {rec0 + PRE_G * REC_DW};
    rowdot1(rec, g0 + PRE_G, g1 - g0 - PRE_G, xq, KA, acc);
    acc[0] += predot(&pre, g0, g1 - g0, xq, KA);
#else
    const __global uint* rec[1] = {rec0};
    rowdot1(rec, g0, g1 - g0, xq, KA, acc);
#endif
    red[sg][ln] = acc[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float v = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int k = 0; k < OPROJ_KS; ++k)
            v += red[sg + k][ln];
        o_out[rb * SG + ln] = v;
    }
}

// ---------------------------------------------------------------------------
// ffn_in: h1 = h + RMS(o)*w_pa, then
//   dense WGs:  a = gelu(gate(x)) * up(x),  x = RMS(h1)*w_pf, written quantised
//   router WGs: logits = Wr . (RMS(h1)*w_r)
// WG 0 also stores h1 and the quantised MoE input RMS(h1)*w_pf2.
// ---------------------------------------------------------------------------
#define HB (H_ / 8)
#define HB_PER_LANE ((HB + LWS - 1) / LWS)

REQD __kernel void ffn_in(const __global float* o, const __global half* h,
                          const __global half* w_pa, const __global half* w_pf,
                          const __global half* w_r, const __global half* w_pf2,
                          const __global uint* gpk, const __global uint* upk,
                          const __global half* rw,
                          __global float* h1_out, __global uint* x2q,
                          __global uint* actq, __global float* logits_out) {
    __local uint xq[XQ_WORDS(H_)];
    __local float xf[H_];
    __local float red[NSG][SG * 2];
    __local float outv[NSG / DENSE_KS * SG];
    __local float tmp[NSG];
    const int wg = get_group_id(0);
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const bool router = wg >= DENSE_WGS;

    // Work-item handles 8-element blocks sg*16 + ln + 256*i.
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
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        const bool ok = b < HB;
        if (router) {
            if (ok)
                vstore8(hv[i] * inv1 * vload_half8(b, w_r), b, xf);
        } else {
            const QB q = quant_block(ok ? hv[i] * inv1 * vload_half8(b, w_pf) : (float8)(0.0f), ok);
            if (ok)
                store_qb_l(q, b, H_, xq);
        }
        if (wg == 0) {
            const QB q = quant_block(ok ? hv[i] * inv1 * vload_half8(b, w_pf2) : (float8)(0.0f), ok);
            if (ok) {
                store_qb_g(q, b, H_, x2q);
                vstore8(hv[i], b, h1_out);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (!router) {
        const int ks = sg % DENSE_KS;
        const int rbl = sg / DENSE_KS;
        const int rb = wg * (NSG / DENSE_KS) + rbl;
        const int gps = (H_ / QG) / DENSE_KS;
        const __global uint* rec[2] = {rec_at(gpk, H_, rb, ks * gps), rec_at(upk, H_, rb, ks * gps)};
        float acc[2];
        rowdot2(rec, ks * gps, gps, xq, H_, acc);
        red[sg][ln] = acc[0];
        red[sg][SG + ln] = acc[1];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (ks == 0) {
            float g = 0.0f, u = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int k = 0; k < DENSE_KS; ++k) {
                g += red[sg + k][ln];
                u += red[sg + k][SG + ln];
            }
            outv[rbl * SG + ln] = gelu_tanh(g) * u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg == 0)
            quant_group_store(outv, wg, I_, actq);
    } else {
        // f16 router weights, one row per lane, K split 16 ways.
        const int n = (wg - DENSE_WGS) * SG + ln;
        const int k0 = sg * (H_ / ROUTER_KS);
        const __global half* row = rw + n * H_ + k0;
        float acc = 0.0f;
        __attribute__((opencl_unroll_hint(4))) for (int c = 0; c < (H_ / ROUTER_KS) / SG; ++c) {
            const float16 w = vload_half16(c, row);
            const float x = as_float(intel_sub_group_block_read((const __local uint*)(xf + k0 + c * SG)));
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < SG; ++j)
                acc = fma(w[j], sub_group_broadcast(x, j), acc);
        }
        red[sg][ln] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg == 0) {
            float v = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int k = 0; k < ROUTER_KS; ++k)
                v += red[k][ln];
            logits_out[n] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// ffn_mid: MoE WGs:  slot t of the top-k: moe_act[t] = gelu(gate_e(x2)) * up_e(x2),
//                    written quantised for ffn_moe_down
//          down WGs: m = down(a)
// The activations come pre-quantised from global memory, so the GEMV starts
// without any SLM prologue or barrier.
// ---------------------------------------------------------------------------
REQD __kernel void ffn_mid(const __global uint* x2q, const __global float* logits, const __global uint* actq,
                           const __global uint* egpk, const __global uint* eupk, const __global uint* dpk,
                           __global uint* moe_actq, __global float* m_out,
                           __global int* tk_idx_out, __global float* tk_w_out) {
    __local float red[NSG][SG * 2];
    __local float outv[NSG / MOE_KS * SG];
    const int wg = get_group_id(0);
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();

    if (wg < MOE_WGS) {
        const int ks = sg % MOE_KS;
        const int rbl = sg / MOE_KS;
        const int slot = wg / MOE_BLK;
        const int blk = wg % MOE_BLK;
        // Every sub-group derives its expert itself: cheaper than a barrier.
        const int expert = topk_softmax(logits, slot, wg == 0 && sg == 0, tk_idx_out, tk_w_out);
        const int nb = blk * (NSG / MOE_KS) + rbl;   // row block within the expert
        const int rb = expert * (IE_ / SG) + nb;
        const int gps = (H_ / QG) / MOE_KS;
        const __global uint* rec[2] = {rec_at(egpk, H_, rb, ks * gps), rec_at(eupk, H_, rb, ks * gps)};
        float acc[2];
        rowdot2g(rec, ks * gps, gps, x2q, H_, acc);
        red[sg][ln] = acc[0];
        red[sg][SG + ln] = acc[1];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (ks == 0) {
            float g = 0.0f, u = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int k = 0; k < MOE_KS; ++k) {
                g += red[sg + k][ln];
                u += red[sg + k][SG + ln];
            }
            outv[rbl * SG + ln] = gelu_tanh(g) * u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg == 0)
            quant_group_store(outv, slot * MOE_BLK + blk, TOPK_ * IE_, moe_actq);
    } else {
        const int ks = sg % DOWN_KS;
        const int rb = (wg - MOE_WGS) * (NSG / DOWN_KS) + sg / DOWN_KS;
        const int g0 = ks * (I_ / QG) / DOWN_KS;
        const int g1 = (ks + 1) * (I_ / QG) / DOWN_KS;
        const __global uint* rec[1] = {rec_at(dpk, I_, rb, g0)};
        float acc[1];
        rowdot1g(rec, g0, g1 - g0, actq, I_, acc);
        red[sg][ln] = acc[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (ks == 0) {
            float v = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int k = 0; k < DOWN_KS; ++k)
                v += red[sg + k][ln];
            m_out[rb * SG + ln] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// ffn_moe_down: e[n] = sum_t w_t * down_{e_t}[n] . moe_act[t]
// Sub-group sg handles slot sg % TOPK_ of row block sg / TOPK_.
// ---------------------------------------------------------------------------
REQD __kernel void ffn_moe_down(const __global uint* moe_actq, const __global int* tk_idx,
                                const __global float* tk_w, const __global uint* edpk,
                                __global float* e_out) {
    __local float red[NSG][SG];
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const int t = sg % TOPK_;
    const int nb = get_group_id(0) * EDOWN_RB + sg / TOPK_;
    const int rb = tk_idx[t] * (H_ / SG) + nb;
    const __global uint* rec[1] = {rec_at(edpk, IE_, rb, 0)};
    float acc[1];
    rowdot1g(rec, t * (IE_ / QG), IE_ / QG, moe_actq, TOPK_ * IE_, acc);
    red[sg][ln] = tk_w[t] * acc[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
        float v = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int k = 0; k < TOPK_; ++k)
            v += red[sg + k][ln];
        e_out[nb * SG + ln] = v;
    }
}

// ---------------------------------------------------------------------------
// ffn_out (one WG): out = (h1 + RMS(RMS(m)*w1 + RMS(e)*w2) * w_post) * layer_scalar
// ---------------------------------------------------------------------------
#define HPT (H_ / LWS)
REQD __kernel void ffn_out(const __global float* h1, const __global float* m, const __global float* e,
                           const __global half* w1, const __global half* w2, const __global half* w_post,
                           const __global half* layer_scalar, __global half* out) {
    __local float tmp[NSG];
    const int lid = get_local_id(0);
    float mv[HPT], ev[HPT];
    float sm = 0.0f, se = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT; ++i) {
        mv[i] = m[lid + i * LWS];
        ev[i] = e[lid + i * LWS];
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

// ---------------------------------------------------------------------------
// attn_qkv: raw = Wqkv . (RMS(x) * w_in), the stacked q|k|v projection rows.
// ---------------------------------------------------------------------------
#define QKV_KS 4                              // 4 row blocks x 4 K slices
REQD __kernel void attn_qkv(const __global half* x, const __global half* w_in, const __global uint* wpk,
                            __global float* raw) {
    __local uint xq[XQ_WORDS(H_)];
    __local float red[NSG][SG];
    __local float tmp[NSG];
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const int ks = sg % QKV_KS;
    const int rb = get_group_id(0) * (NSG / QKV_KS) + sg / QKV_KS;
    const int g0 = ks * (H_ / QG) / QKV_KS;
    const int g1 = (ks + 1) * (H_ / QG) / QKV_KS;
    const __global uint* rec0 = rec_at(wpk, H_, rb, g0);
#if PRE_G
    const PreRec pre = preload(rec0, g1 - g0);
#endif

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
        const bool ok = b < HB;
        const QB q = quant_block(ok ? hv[i] * inv * vload_half8(b, w_in) : (float8)(0.0f), ok);
        if (ok)
            store_qb_l(q, b, H_, xq);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[1];
#if PRE_G
    const __global uint* rec[1] = {rec0 + PRE_G * REC_DW};
    rowdot1(rec, g0 + PRE_G, g1 - g0 - PRE_G, xq, H_, acc);
    acc[0] += predot(&pre, g0, g1 - g0, xq, H_);
#else
    const __global uint* rec[1] = {rec0};
    rowdot1(rec, g0, g1 - g0, xq, H_, acc);
#endif
    red[sg][ln] = acc[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float v = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int k = 0; k < QKV_KS; ++k)
            v += red[sg + k][ln];
        raw[rb * SG + ln] = v;
    }
}

// ---------------------------------------------------------------------------
// attn_post: one sub-group per head. q and k heads: RMS * gamma, then
// rotate-half RoPE over the leading `rn` dims; v heads: gamma-less RMS.
// meta = {HQ, HK, D, rn, raw k row, raw v row, raw q row, position stride
// (ints), position rounded to half, table width C}. The RoPE tables are those of
// the exported rotary_emb: cos/sin(half(pos) * inv_freq[i % (C/2)]) in f32,
// stored as half. out is head-major [B, HQ+2*HK, S, D]; t = b * S + s.
// Token t = t0 + get_group_id(1) reads raw + get_group_id(1) * rstride.
// ---------------------------------------------------------------------------
#define ATTN_DMAX 512
REQD __kernel void attn_post(const __global float* raw, const __global half* gq, const __global half* gk,
                             const __global int* meta, const __global int* pos, const __global float* invf,
                             __global half* out_base, const int t0, const int S, const int rstride) {
    __local float sv[NSG][ATTN_DMAX];
    __local float ctab[ATTN_DMAX], stab[ATTN_DMAX];
    const int t = t0 + get_group_id(1);
    raw += (size_t)get_group_id(1) * rstride;
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const int head = get_group_id(0) * NSG + sg;
    const int HQ = meta[0], HK = meta[1], D = meta[2], RN = meta[3];
    if (get_group_id(0) * NSG >= HQ + 2 * HK)
        return;
    {
        const int half_c = meta[9] / 2;
        float p = (float)pos[(size_t)t * meta[7]];
        if (meta[8])
            p = (float)convert_half_rte(p);
        for (int i = get_local_id(0); i < RN; i += LWS) {
            const float f = p * invf[i % half_c];
            ctab[i] = (float)convert_half_rte(cos(f));
            stab[i] = (float)convert_half_rte(sin(f));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (head >= HQ + 2 * HK)
        return;
    const int HT = HQ + 2 * HK;
    __global half* out = out_base + ((size_t)((t / S) * HT + head) * S + t % S) * D;
    const int dst = 0;
    const __global half* g = 0;
    if (head < HQ) {
        raw += meta[6] + head * D;
        g = gq;
    } else if (head < HQ + HK) {
        raw += meta[4] + (head - HQ) * D;
        g = gk;
    } else {
        raw += meta[5] + (head - HQ - HK) * D;
    }
    const int nj = D / SG;
    float ss = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int j = 0; j < ATTN_DMAX / SG; ++j) {
        if (j < nj) {
            const float v = raw[j * SG + ln];
            sv[sg][j * SG + ln] = v;
            ss += v * v;
        }
    }
    const float r = rsqrt(sub_group_reduce_add(ss) / D + EPS_);
    if (g) {
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < ATTN_DMAX / SG; ++j)
            if (j < nj)
                sv[sg][j * SG + ln] *= r * vload_half(j * SG + ln, g);
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        const int half_rn = RN / 2;
        for (int i = ln; i < D; i += SG) {
            float y = sv[sg][i];
            if (i < half_rn)
                y = y * ctab[i] - sv[sg][i + half_rn] * stab[i];
            else if (i < RN)
                y = y * ctab[i] + sv[sg][i - half_rn] * stab[i];
            vstore_half(y, dst + i, out);
        }
    } else {
        for (int i = ln; i < D; i += SG)
            vstore_half(sv[sg][i] * r, dst + i, out);
    }
}

// ---------------------------------------------------------------------------
// Attention masks [B,1,S,L] from attention_mask [B,L] and token_type_ids [B,S]
// (i32 or i64: mm[8] / mm[9] ints per element). Query i sits at L - S + i.
// mm = {fill, -, pad_mul, pad_add, window, causal_off, vision_id0, vision_id1}:
//   full    = (j >= q + causal_off ? fill : 0) + am[j] * pad_mul + pad_add
//   sliding = same vision group ? 0 : (q - j >= window ? fill : full)
// Vision groups number the runs of vision tokens. As exported, token types are
// padded on the right to L, so positions >= S count as text.
// ---------------------------------------------------------------------------
inline int vis_at(const __global int* tti, int istride, int S, int p, float v0, float v1) {
    if (p >= S)
        return 0;
    const float t = (float)tti[(size_t)p * istride];
    return t == v0 || t == v1;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void mask_groups(const __global int* tti, const __global float* mm, const int S, const int L,
                          __global int* grp) {
    __local int carry;
    const int istride = (int)mm[9];
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    tti += (size_t)b * S * istride;
    grp += (size_t)b * L;
    if (lid == 0)
        carry = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p0 = 0; p0 < L; p0 += 256) {
        const int p = p0 + lid;
        const int vis = p < L ? vis_at(tti, istride, S, p, mm[6], mm[7]) : 0;
        const int prev = p > 0 && p - 1 < L ? vis_at(tti, istride, S, p - 1, mm[6], mm[7]) : 0;
        const int base = carry;
        const int run = work_group_scan_inclusive_add(vis & !prev);
        if (p < L)
            grp[p] = vis ? base + run - 1 : -1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 255)
            carry = base + run;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void mask_fill(const __global int* am, const __global float* mm, const __global int* grp, const int S,
                        const int L, const int sliding, __global half* out) {
    const int istride = (int)mm[8];
    const int j = get_global_id(0);
    const int i = get_global_id(1);
    const int b = get_global_id(2);
    if (j >= L)
        return;
    const int q = L - S + i;
    const float a = (float)am[((size_t)b * L + j) * istride];
    float v = (j >= q + (int)mm[5] ? mm[0] : 0.0f) + fma(a, mm[2], mm[3]);
    if (sliding) {
        const int gj = grp[(size_t)b * L + j];
        if (gj >= 0 && gj == grp[(size_t)b * L + q])
            v = 0.0f;
        else if (q - j >= (int)mm[4])
            v = mm[0];
    }
    vstore_half_rte(v, ((size_t)b * S + i) * L + j, out);
}
)CLC";

constexpr int kAttnQkvRowsPerWg = 64;  // SG * NSG / QKV_KS
