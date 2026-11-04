#pragma once

#include <cstddef>
#include <cstdint>

// Whole-model Gemma-4 MegaKernel (kModel, see gemma4POCParams.h).
//
// Decode (one token) is ONE persistent launch, mk_decode: NWG co-resident
// work-groups walk all 30 decoder layers and lm_head, separated by grid-wide
// barriers (atomic counters in device memory). Per layer:
//   A  x = ffn_out(prev layer); q|k|v = Wqkv . (RMS(x) * w_in)
//   B  q/k norm + RoPE, KV-cache append, split-K flash decoding
//   C  o = o_proj(attention)
//   D  h1 = x + RMS(o) w_pa; dense gate/up; router logits
//   E  MoE gate/up (top-k experts); dense down
//   F  MoE down (weighted by the routing weights)
// and finally G: x = ffn_out(layer 29), lm_head (u8 per-row) + logit soft-cap.
//
// Every int4 GEMV phase is a flat stream of repacked records (gemma4Repack.h):
// the records of all matrices of a phase are concatenated and each sub-group of
// the grid takes an equal contiguous slice, so all NWG * 16 sub-groups stream
// the same number of bytes. A slice may start or end inside a row block, so
// row sums are combined with float atomics. Accumulators are zeroed two phases
// after their last reader (see the zeroing notes in mk_decode).
//
// Prompt calls (several tokens) instead run the batched kernels of the per-layer
// runtimes plus the helpers at the end of this file.
//
// Appended to kFfnSource; built with its -D flags plus NL_, NWG, V_, RMAX_.
static const char* kModelSource = R"CLC(
#ifndef NWG
#define NWG 80
#endif
#define NSGT (NWG * NSG)
#define ATT_DMAX 512
#define NPH 6                          // grid barriers per layer
#define ATT_QD 512                     // q elements one sub-group keeps in registers (GPS * D)
#ifndef ATT_KB
#define ATT_KB 1                       // keys whose K / V loads are in flight together
#endif
#ifndef SPIN_PAUSE
#define SPIN_PAUSE 8
#endif
#ifndef SPIN_MAX
#define SPIN_MAX (1 << 24)
#endif
#ifndef PF_RECS
#define PF_RECS 4                      // records per sub-group prefetched across a grid barrier
#endif
#define ROUTER_WG0 (NWG - E_ / SG)
#ifndef OPROJ_PF_MAX
#define OPROJ_PF_MAX 12000             // records of o_proj prefetched during attention (~6.9 MB)
#endif     // phase D: the last E_/SG work-groups compute the router

extern void __builtin_IB_eu_thread_pause(uint);

// Host mirror: ModelLayerTab in gemma4ModelRuntime.cpp (same member order).
typedef struct {
    const __global half* w_in;
    const __global uint* qkv;
    const __global half* gq;
    const __global half* gk;
    const __global float* invf;
    const __global uint* oproj;
    const __global half* w_pa;
    const __global half* w_pf;
    const __global half* w_r;
    const __global half* w_pf2;
    const __global uint* gate;
    const __global uint* up;
    const __global uint* down;
    const __global half* rw;
    const __global uint* egate;
    const __global uint* eup;
    const __global uint* edown;
    const __global half* w_pf1;
    const __global half* w_pf2p;
    const __global half* w_post;
    const __global half* ls;
    __global half* kc;                 // [HK][cap][D]
    __global half* vc;
    int R, HQ, HK, D, RN, q_off, k_off, v_off, half_c, pos_f16, sliding, pad;
} LayerTab;

typedef struct {
    __global half* xbuf;               // [H] residual entering the current layer (f16, as the graph)
    __global long* raw;                // [RMAX_] q|k|v projections (accumulators: fixed point, see atomic_addfx)
    __global half* att;                // [8192] attention output
    __global long* o;                  // [H] o_proj
    __global float* h1;                // [H]
    __global long* gu;                 // [2 * I_] dense gate | up
    __global long* lg;                 // [E_] router logits
    __global long* egu;                // [TOPK_][2][IE_] expert gate | up
    __global long* m;                  // [H] dense down
    __global long* e;                  // [H] MoE down
    __global float* part;              // attention partials
    __global int* sync;                // grid barrier counters
    __global int* tick;                // [NL_ * 8] attention combine tickets
    __global int* err;
    const __global half* w_final;
    const __global uint* lm;           // u8 lm_head records (16 rows x 64 bytes)
    const __global half* lm_s;         // [V_]
    const __global uchar* lm_z;        // [V_]
    float cap_div, cap_mul, attn_scale, window;
    float causal_off, p_mul, p_add, pad0;
    int pos_stride, am_stride, kv_cap, pad1;
    __global ulong* prof;              // MK_PROF: [barrier][NWG][arrive, release, ...]
    __global uint* xqm;                // XQ_WORDS(TOPK_ * IE_) quantised expert activations
    __global uint* xqd;                // XQ_WORDS(I_) quantised dense activation
} ModelTab;

extern ulong __builtin_IB_read_cycle_counter(void);
#ifdef MK_PROF
#define PROF_MARK(mt, bar, which)                                                                   \
    if (get_local_id(0) == 0)                                                                       \
        (mt)->prof[((size_t)(bar) * NWG + get_group_id(0)) * 5 + (which)] = __builtin_IB_read_cycle_counter();
#define PROF_DBG(mt, layer, u, point)                                                               \
    if (get_local_id(0) == 0 && (u) == 0)                                                           \
        (mt)->prof[(size_t)(NL_ * NPH + 2) * NWG * 5 + (layer) * 16 + (point)] = __builtin_IB_read_cycle_counter();
#else
#define PROF_MARK(mt, bar, which)
#define PROF_DBG(mt, layer, u, point)
#endif

// ---------------------------------------------------------------------------
// Grid barrier: every work-group increments the counter once; waiters spin
// until it reaches epoch * NWG. Counters are never reset, the epoch grows by
// one per launch. A bounded spin turns a co-residency failure into garbage
// output plus an error flag instead of a GPU hang.
// ---------------------------------------------------------------------------
inline void grid_arrive(__global int* c) {
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
        atomic_fetch_add_explicit((volatile __global atomic_int*)c, 1, memory_order_release, memory_scope_device);
}

inline void grid_wait(__global int* c, int target, __global int* err) {
    if (get_local_id(0) == 0) {
        int n = 0;
        while (atomic_load_explicit((volatile __global atomic_int*)c, memory_order_relaxed, memory_scope_device) <
               target) {
            __builtin_IB_eu_thread_pause(SPIN_PAUSE);
            if (++n > SPIN_MAX) {
                *err = 1;
                break;
            }
        }
        atomic_load_explicit((volatile __global atomic_int*)c, memory_order_acquire, memory_scope_device);
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

// Split-K sums are accumulated as Q32.32 fixed point with 64-bit integer
// atomics: integer addition is associative, so the result is the exact sum of
// the (float) partials whatever order the sub-groups arrive in - unlike float
// atomics, which made every run differ slightly and flipped near-tied MoE
// experts. |sum| must stay below 2^31.
#define FX_ONE 4294967296.0f
inline void atomic_addfx(__global long* p, float v) {
    atomic_fetch_add_explicit((volatile __global atomic_long*)p, convert_long_rte(v * FX_ONE), memory_order_relaxed,
                              memory_scope_device);
}
inline float fx2f(long v) {
    return (float)v * (1.0f / FX_ONE);
}
inline float8 fx2f8(long8 v) {
    return convert_float8(v) * (1.0f / FX_ONE);
}

inline float8 gelu_tanh8(float8 x) {
    return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// topk_softmax with the result in SLM; called by one sub-group.
inline void topk_local(const __global long* logits, __local int* tk_idx, __local float* tk_w) {
    const int ln = get_sub_group_local_id();
    float v[E_ / SG];
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < E_ / SG; ++i) {
        const float l = fx2f(logits[ln + SG * i]);
        v[i] = isnan(l) ? -INFINITY : l;
    }
    float top = 0.0f, wsum = 0.0f, mine = 0.0f;
    int mine_idx = 0;
    __attribute__((opencl_unroll_hint)) for (int t = 0; t < TOPK_; ++t) {
        float lm = v[0];
        int li = ln;
        __attribute__((opencl_unroll_hint)) for (int i = 1; i < E_ / SG; ++i) {
            if (v[i] > lm) {
                lm = v[i];
                li = ln + SG * i;
            }
        }
        const float mxv = sub_group_reduce_max(lm);
        const int idx = sub_group_reduce_min(lm == mxv ? li : 0x7fffffff);
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < E_ / SG; ++i)
            if (ln + SG * i == idx)
                v[i] = -INFINITY;
        if (t == 0)
            top = mxv;
        const float e = exp(mxv - top);
        wsum += e;
        if (ln == t) {
            mine = e;
            mine_idx = idx;
        }
    }
    if (ln < TOPK_) {
        tk_idx[ln] = mine_idx;
        tk_w[ln] = mine / wsum;
    }
}

inline void zero_grid(__global long* p, int n) {
    for (int i = get_group_id(0) * LWS + get_local_id(0); i < n; i += NWG * LWS)
        p[i] = 0;
}

inline void prefetch_recs(const __global uint* rec, int n) {
    for (int i = 0; i < n; ++i) {
        intel_sub_group_block_prefetch_ui8(rec + i * REC_DW);
        intel_sub_group_block_prefetch_ui(rec + i * REC_DW + 128);
    }
}

// The contiguous record slice [r0, r1) of `total` records owned by sub-group gs of ns.
inline int slice_lo(int total, int gs, int ns) {
    return (int)(((long)total * gs) / ns);
}

// ---------------------------------------------------------------------------
// Flat record streaming over a job list (int4 GEMV phases)
// ---------------------------------------------------------------------------
typedef struct {
    const __global uint* m;            // records of the job's first row block
    __global long* out;                // row r of the job -> out[r]
    int G;                             // groups per row block
    int nrec;                          // records of the job
    int xq;                            // SLM word offset of the quantised activation
    int K;                             // its length
    int gx;                            // activation group of the job's group 0
    float wmul;
} Job;

// One group of a record against the activation group g.
#define GROUP_DOT(wv, meta, g, xh, xl, xa, xs, acc)                                                     \
    {                                                                                                 \
        const uint xhv_ = intel_sub_group_block_read(xh + (g) * 16);                                  \
        const uint xlv_ = intel_sub_group_block_read(xl + (g) * 16);                                  \
        int ih = 0, il = 0;                                                                           \
        DOT8(wv, xhv_, xlv_, ih, il)                                                                  \
        const float sc_ = (float)as_half2(meta).s0;                                                   \
        const float zz_ = (float)((meta) >> 16);                                                      \
        acc = fma(sc_, fma(xa[g], fma((float)il, 1.0f / XL_RES, (float)ih), -zz_ * xs[g]), acc);     \
    }

#ifndef SEG_BATCH
#define SEG_BATCH 4
#endif
// acc over groups [g0, g0 + ng) of one row block; rec = record of group g0.
// Records are loaded SEG_BATCH at a time, every load before the first use.
inline float seg_dot(const __global uint* rec, const int g0, const int ng, const __local uint* xq, const int K) {
    const __local uint* xh = xq;
    const __local uint* xl = xq + K / 4;
    const __local float* xa = (const __local float*)(xq + K / 2);
    const __local float* xs = xa + K / QG;
    float acc = 0.0f;
    int gi = 0;
    for (; gi + SEG_BATCH <= ng; gi += SEG_BATCH) {
        uint8 wv[SEG_BATCH];
        uint mv[SEG_BATCH];
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < SEG_BATCH; ++b) {
            wv[b] = intel_sub_group_block_read8(rec + (gi + b) * REC_DW);
            mv[b] = intel_sub_group_block_read(rec + (gi + b) * REC_DW + 128);
        }
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < SEG_BATCH; ++b)
            GROUP_DOT(wv[b], mv[b], g0 + gi + b, xh, xl, xa, xs, acc)
    }
    for (; gi < ng; ++gi) {
        const uint8 wv = intel_sub_group_block_read8(rec + gi * REC_DW);
        const uint m_ = intel_sub_group_block_read(rec + gi * REC_DW + 128);
        GROUP_DOT(wv, m_, g0 + gi, xh, xl, xa, xs, acc)
    }
    return acc;
}

// Records [r, r1) of the concatenated jobs.
inline void run_jobs_range(const __local Job* jobs, const int njobs, int r, const int r1, const __local uint* slm) {
    int j = 0, base = 0;
    while (j < njobs && r >= base + jobs[j].nrec) {
        base += jobs[j].nrec;
        ++j;
    }
    const int ln = get_sub_group_local_id();
    while (r < r1) {
        const Job jb = jobs[j];
        const int lr = r - base;
        const int rb = lr / jb.G;
        const int g0 = lr - rb * jb.G;
        const int end = min(r1 - base, (rb + 1) * jb.G);
        const int ng = end - lr;
        const float acc = seg_dot(jb.m + (size_t)lr * REC_DW, jb.gx + g0, ng, slm + jb.xq, jb.K);
        atomic_addfx(jb.out + rb * SG + ln, acc * jb.wmul);
        r += ng;
        if (r - base >= jb.nrec) {
            base += jb.nrec;
            ++j;
        }
    }
}

// Sub-group gs of ns streams its slice of the concatenated jobs.
inline void run_jobs(const __local Job* jobs, const int njobs, const int gs, const int ns, const __local uint* slm) {
    int total = 0;
    for (int j = 0; j < njobs; ++j)
        total += jobs[j].nrec;
    run_jobs_range(jobs, njobs, slice_lo(total, gs, ns), slice_lo(total, gs + 1, ns), slm);
}

// First record of sub-group gs's slice of a single-matrix phase.
inline void prefetch_single(const __global uint* m, int total, int gs, int ns) {
    const int r0 = slice_lo(total, gs, ns), r1 = slice_lo(total, gs + 1, ns);
    prefetch_recs(m + (size_t)r0 * REC_DW, min(PF_RECS, r1 - r0));
}

// ---------------------------------------------------------------------------
// Element-wise prologues. Work-item lid owns elements lid + LWS * i.
// ---------------------------------------------------------------------------
#define HPT_ (H_ / LWS)

// Residual leaving layer P (ffn_out of the per-layer kernels), rounded to f16
// as the graph stores it.
inline void ffn_out_vals(const __global LayerTab* P, const __global ModelTab* mt, float xv[HPT_], __local float* tmp) {
    const int lid = get_local_id(0);
    const __global long* m = mt->m;
    const __global long* e = mt->e;
    const __global float* h1 = mt->h1;
    float mv[HPT_], ev[HPT_];
    float sm = 0.0f, se = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
        mv[i] = fx2f(m[lid + i * LWS]);
        ev[i] = fx2f(e[lid + i * LWS]);
        sm += mv[i] * mv[i];
        se += ev[i] * ev[i];
    }
    const float inv_m = rsqrt(wg_sum(sm, tmp) / H_ + EPS_);
    const float inv_e = rsqrt(wg_sum(se, tmp) / H_ + EPS_);
    float ss = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
        const int k = lid + i * LWS;
        mv[i] = mv[i] * inv_m * vload_half(k, P->w_pf1) + ev[i] * inv_e * vload_half(k, P->w_pf2p);
        ss += mv[i] * mv[i];
    }
    const float inv_s = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
    const float ls = vload_half(0, P->ls);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
        const int k = lid + i * LWS;
        xv[i] = (float)convert_half((h1[k] + mv[i] * inv_s * vload_half(k, P->w_post)) * ls);
    }
}

// dst[k] = v[k] * RMS(v) * gamma[k] into SLM.
inline void rms_to_slm(const float v[HPT_], const __global half* gamma, __local float* dst, __local float* tmp) {
    const int lid = get_local_id(0);
    float ss = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
        ss += v[i] * v[i];
    const float inv = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
        const int k = lid + i * LWS;
        dst[k] = v[i] * inv * vload_half(k, gamma);
    }
}

// Quantises K floats from SLM into the GEMV activation layout (whole WG).
inline void quant_slm(const __local float* src, const int K, __local uint* xq) {
    for (int c0 = get_sub_group_id() * SG; c0 < K / 8; c0 += NSG * SG) {
        const int b = c0 + get_sub_group_local_id();
        const bool ok = b < K / 8;
        const QB q = quant_block(ok ? vload8(b, src) : (float8)(0.0f), ok);
        if (ok)
            store_qb_l(q, b, K, xq);
    }
}

// ---------------------------------------------------------------------------
// Phase B: attention for one (kv head, key split) unit.
// Each sub-group keeps GPS = ATT_QD / D query heads in registers (strided
// layout: element d = j * 16 + lane) and walks every NKL-th key of the split.
// ---------------------------------------------------------------------------
// Key range [j0, j1) of split `split` out of NS over the visible keys [jlo, jhi].
inline int2 split_keys(const int jlo, const int jhi, const int NS, const int split) {
    const int chunk = (jhi - jlo + NS) / NS;
    const int j0 = jlo + split * chunk;
    return (int2)(j0, min(jhi + 1, j0 + chunk));
}

// Head vector of one lane (element d = j * 16 + lane), RMS-normalised with
// gamma (or without when g is null), then rotate-half RoPE over d < RN. The
// partner of element j is j +- RN / 32 of the same lane. All loads are issued
// before the first use.
// Loads use sub-group block reads: one uniform address returns exactly the
// lane-strided layout, instead of a 64-bit address per lane and element.
#define HEAD_NORM_ROPE(OUT, OFF, NJ, RNL, r, g, cs, sn)                                                    \
    {                                                                                                     \
        float ss_ = 0.0f;                                                                                 \
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < (NJ) / 8; ++c) {                          \
            const float8 v_ = fx2f8(as_long8(intel_sub_group_block_read_ul8((const __global ulong*)(r) + c * 128))); \
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i) {                             \
                OUT[(OFF) + c * 8 + i] = v_[i];                                                           \
                ss_ = fma(v_[i], v_[i], ss_);                                                             \
            }                                                                                             \
        }                                                                                                 \
        const float inv_ = rsqrt(sub_group_reduce_add(ss_) / (NJ * SG) + EPS_);                          \
        if (g) {                                                                                          \
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < (NJ) / 8; ++c) {                      \
                const half8 g_ = as_half8(intel_sub_group_block_read_us8((const __global ushort*)(g) + c * 128)); \
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)                           \
                    OUT[(OFF) + c * 8 + i] *= inv_ * (float)g_[i];                                        \
            }                                                                                             \
            const int hr_ = (RNL) / SG / 2;                                                               \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < hr_; ++j) {                           \
                const float a_ = OUT[(OFF) + j], b_ = OUT[(OFF) + j + hr_];                               \
                const int d_ = j * SG + ln, e_ = d_ + hr_ * SG;                                           \
                OUT[(OFF) + j] = a_ * cs[d_] - b_ * sn[d_];                                               \
                OUT[(OFF) + j + hr_] = b_ * cs[e_] + a_ * sn[e_];                                         \
            }                                                                                             \
        } else {                                                                                          \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < (NJ); ++j)                            \
                OUT[(OFF) + j] *= inv_;                                                                   \
        }                                                                                                 \
    }

// D, RN and GPS are literals at both call sites, so after inlining every
// register array is statically indexed. Each sub-group keeps GPS = 512 / D
// query heads and walks every NKL-th key of the split, KB = GPS keys per
// batch with all of their loads in flight together.
__attribute__((always_inline)) inline void attn_unit(const __global LayerTab* L, const __global ModelTab* mt, const int layer, const int u,
                      const int NS, const int past, const int jlo, const int jhi, const float posf,
                      const __global int* am, __local float* slm, const int att_base, const int D,
                      const int RN) {
    const int GPS = ATT_QD / D, KB = ATT_KB;
    const int nj = D / SG;
    const int HQ = L->HQ, HK = L->HK;
    const int G = HQ / HK;
    const int QSPLIT = G / GPS;
    const int NKL = NSG / QSPLIT;
    const int kvh = u / NS, split = u % NS;
    const int sg = get_sub_group_id(), ln = get_sub_group_local_id(), lid = get_local_id(0);
    const int qset = sg % QSPLIT, kl = sg / QSPLIT;
    const int cap = mt->kv_cap;
    const int half_c = L->half_c;
    const __global float* invf = L->invf;
    const __global half* gq = L->gq;
    const __global half* gk = L->gk;
    const __global long* raw = mt->raw;
    const int q_off = L->q_off, k_off = L->k_off, v_off = L->v_off;
    __global half* kc = L->kc + (size_t)kvh * cap * D;
    __global half* vc = L->vc + (size_t)kvh * cap * D;
    __local float* slm_acc = slm;                          // [NSG][ATT_QD]
    __local float* msg = slm + NSG * ATT_QD;               // [NSG][2]
    __local float* lsm = msg + 2 * NSG;                    // [NSG][2]
    __local float* wt = lsm + 2 * NSG;                     // [G][NS]
    __local half* kvcur = (__local half*)(wt + 8 * 64);    // [2][D]
    __local int* flag = (__local int*)(kvcur + 2 * ATT_DMAX);
    PROF_DBG(mt, layer, u, 0)

    const int2 jr = split_keys(jlo, jhi, NS, split);
    const int j0 = jr.x, j1 = jr.y;
#ifdef ATT_STUB
    return;
#endif

    // RoPE tables (one sin / cos per work-item), read per lane from SLM.
    __local float* cs = (__local float*)(flag + 4);
    __local float* sn = cs + ATT_DMAX;
    for (int i = lid; i < RN; i += LWS) {
        const float f = posf * invf[i % half_c];
        cs[i] = (float)convert_half_rte(cos(f));
        sn[i] = (float)convert_half_rte(sin(f));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#if ATT_STOP == 1
    mt->att[lid] = cs[lid] + sn[lid];
    return;
#endif
    // The current token's k / v: appended by the split that owns it, which
    // also reads them back from SLM instead of the cache.
    const bool own_cur = j1 == jhi + 1 && past <= jhi;
    if (own_cur && sg < 2) {
        float hv[ATT_DMAX / SG];
        const __global long* hr = raw + (sg == 0 ? k_off : v_off) + kvh * D;
        const __global half* hg = sg == 0 ? gk : (const __global half*)0;
        HEAD_NORM_ROPE(hv, 0, nj, RN, hr, hg, cs, sn)
        __global ushort* dst = (__global ushort*)((sg == 0 ? kc : vc) + (size_t)past * D);
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c) {
            ushort8 h;
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                h[i] = as_ushort(convert_half(hv[c * 8 + i]));
            intel_sub_group_block_write_us8(dst + c * 128, h);
            intel_sub_group_block_write_us8((__local ushort*)kvcur + sg * ATT_DMAX + c * 128, h);
        }
    }
#if ATT_STOP == 2
    return;
#endif
    float qv[ATT_QD / SG], acc[ATT_QD / SG];
    float mx[2], ls[2];
    __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
        const int qh = kvh * G + qset * GPS + hh;
        HEAD_NORM_ROPE(qv, hh * nj, nj, RN, raw + q_off + qh * D, gq, cs, sn)
        // q is f16 in the graph (RoPE output); the rounding matters for the
        // peaked full-attention heads.
        const float qs = mt->attn_scale;
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < nj; ++j)
            qv[hh * nj + j] = (float)convert_half(qv[hh * nj + j]) * qs;
        mx[hh] = -INFINITY;
        ls[hh] = 0.0f;
    }
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < ATT_QD / SG; ++i)
        acc[i] = 0.0f;
#ifdef ATT_NO_SLM_CUR
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#else
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    PROF_DBG(mt, layer, u, 3)
#if ATT_STOP == 3
    { float z = 0; for (int i = 0; i < 32; ++i) z += qv[i]; mt->att[lid] = z; return; }
#endif

    const int ams = mt->am_stride;
    const float p_mul = mt->p_mul, p_add = mt->p_add;
    for (int t0 = j0 + kl; t0 < j1; t0 += NKL * KB) {
        half8 kk[2][ATT_DMAX / SG / 8], vv[2][ATT_DMAX / SG / 8];
        float madd[2];
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < KB; ++b) {
            const int j = t0 + b * NKL;
            madd[b] = -INFINITY;
            if (j < j1) {
                madd[b] = am ? (float)convert_half_rte(fma((float)am[(size_t)j * ams], p_mul, p_add)) : 0.0f;
#ifdef ATT_NO_SLM_CUR
                const bool cur = false;
#else
                const bool cur = j == past;
#endif
                const __global ushort* kp = (const __global ushort*)(kc + (size_t)j * D);
                const __global ushort* vp = (const __global ushort*)(vc + (size_t)j * D);
                const __local ushort* kl_ = (const __local ushort*)kvcur;
                __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c) {
                    kk[b][c] = as_half8(cur ? intel_sub_group_block_read_us8(kl_ + c * 128)
                                            : intel_sub_group_block_read_us8(kp + c * 128));
                    vv[b][c] = as_half8(cur ? intel_sub_group_block_read_us8(kl_ + ATT_DMAX + c * 128)
                                            : intel_sub_group_block_read_us8(vp + c * 128));
                }
            }
        }
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < KB; ++b) {
            if (madd[b] < -1e4f)
                continue;
            float s[2] = {0.0f, 0.0f};
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c)
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                    __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh)
                        s[hh] = fma(qv[hh * nj + c * 8 + i], (float)kk[b][c][i], s[hh]);
            float pr[2], cr[2];
            __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
                const float sc = sub_group_reduce_add(s[hh]) + madd[b];
                const float mn = fmax(mx[hh], sc);
                cr[hh] = exp(mx[hh] - mn);
                pr[hh] = exp(sc - mn);
                ls[hh] = fma(ls[hh], cr[hh], pr[hh]);
                mx[hh] = mn;
            }
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c)
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                    __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
                        const int k = hh * nj + c * 8 + i;
                        acc[k] = fma(acc[k], cr[hh], pr[hh] * (float)vv[b][c][i]);
                    }
        }
    }
    PROF_DBG(mt, layer, u, 4)
#if ATT_STOP == 4
    { float z = mx[0] + ls[0]; for (int i = 0; i < 32; ++i) z += acc[i]; mt->att[lid] = z; return; }
#endif

    // Work-group combine over the NKL key lanes of each head: rescale to the
    // common maximum, then a tree through SLM.
    if (ln == 0) {
        __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh)
            msg[sg * 2 + hh] = mx[hh];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float M[2];
    __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
        M[hh] = -INFINITY;
        for (int k = 0; k < NKL; ++k)
            M[hh] = fmax(M[hh], msg[(k * QSPLIT + qset) * 2 + hh]);
        const float w = mx[hh] == -INFINITY ? 0.0f : exp(mx[hh] - M[hh]);
        ls[hh] *= w;
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < nj; ++j)
            acc[hh * nj + j] *= w;
    }
    for (int half_ = NKL / 2; half_ >= 1; half_ /= 2) {
        if (kl >= half_ && kl < 2 * half_) {
            const int slot = (kl - half_) * QSPLIT + qset;
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < ATT_QD / SG; ++i)
                slm_acc[slot * ATT_QD + i * SG + ln] = acc[i];
            if (ln == 0) {
                lsm[slot * 2] = ls[0];
                lsm[slot * 2 + 1] = ls[1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kl < half_) {
            const int slot = kl * QSPLIT + qset;
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < ATT_QD / SG; ++i)
                acc[i] += slm_acc[slot * ATT_QD + i * SG + ln];
            ls[0] += lsm[slot * 2];
            ls[1] += lsm[slot * 2 + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int PS = D + 16;  // keeps partials 64-byte aligned
    // A single split is final already.
    if (NS == 1) {
        if (kl == 0) {
            __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
                const int qh = kvh * G + qset * GPS + hh;
                const float il = 1.0f / ls[hh];
                __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c) {
                    ushort8 h;
                    __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                        h[i] = as_ushort(convert_half(acc[hh * nj + c * 8 + i] * il));
                    intel_sub_group_block_write_us8((__global ushort*)(mt->att + qh * D) + c * 128, h);
                }
            }
        }
        PROF_DBG(mt, layer, u, 5)
        PROF_DBG(mt, layer, u, 6)
        return;
    }
    if (kl == 0) {
        __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
            const int qh = kvh * G + qset * GPS + hh;
            __global float* pp = mt->part + ((size_t)qh * NS + split) * PS;
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c)
                intel_sub_group_block_write8((__global uint*)(pp + c * 128), as_uint8((float8)(
                    acc[hh * nj + c * 8], acc[hh * nj + c * 8 + 1], acc[hh * nj + c * 8 + 2],
                    acc[hh * nj + c * 8 + 3], acc[hh * nj + c * 8 + 4], acc[hh * nj + c * 8 + 5],
                    acc[hh * nj + c * 8 + 6], acc[hh * nj + c * 8 + 7])));
            if (ln == 0) {
                pp[D] = M[hh];
                pp[D + 1] = ls[hh];
            }
        }
    }
    PROF_DBG(mt, layer, u, 5)
#ifdef ATT_SKIP_COMBINE
    return;
#endif

    // All NS units of this kv head wait for each other, then each merges a
    // 1 / NS share of the G * D outputs. The counter grows by NS per launch;
    // att_base is its value before this launch.
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    __global int* t = mt->tick + layer * 8 + kvh;
    grid_arrive(t);
    grid_wait(t, att_base + NS, mt->err);
    const int chunk = (G * D + NS - 1) / NS;
    const int o0 = split * chunk, o1 = min(G * D, o0 + chunk);
    const int h0 = o0 / D, h1 = (o1 - 1) / D;
    // Per (head, split) weights exp(m - M) / L: gather m, l once, reduce per head.
    const int nh = h1 - h0 + 1;
    __local float* mm = slm_acc;
    __local float* ll = slm_acc + 8 * 64;
    __local float* hml = slm_acc + 16 * 64;
    for (int i = lid; i < nh * NS; i += LWS) {
        const __global float* pp = mt->part + ((size_t)(kvh * G + h0 + i / NS) * NS + i % NS) * PS + D;
        mm[i] = pp[0];
        ll[i] = pp[1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < nh) {
        float Mx = -INFINITY, Lsum = 0.0f;
        for (int p = 0; p < NS; ++p)
            Mx = fmax(Mx, mm[lid * NS + p]);
        for (int p = 0; p < NS; ++p)
            Lsum = fma(exp(mm[lid * NS + p] - Mx), ll[lid * NS + p], Lsum);
        hml[2 * lid] = Mx;
        hml[2 * lid + 1] = 1.0f / Lsum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < nh * NS; i += LWS)
        wt[i] = exp(mm[i] - hml[2 * (i / NS)]) * hml[2 * (i / NS) + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = o0 + lid; o < o1; o += LWS) {
        const int hw = o / D, d = o - hw * D;
        const __global float* pb = mt->part + (size_t)(kvh * G + hw) * NS * PS + d;
        float a = 0.0f;
        __attribute__((opencl_unroll_hint(8))) for (int p = 0; p < NS; ++p)
            a = fma(wt[(hw - h0) * NS + p], pb[p * PS], a);
        mt->att[(kvh * G + hw) * D + d] = convert_half(a);
    }
    PROF_DBG(mt, layer, u, 6)
}

// Separate (non-inlined) instantiations keep the attention's register
// pressure out of the main loop of mk_decode.
#ifdef ATT_NOINLINE
#define ATT_FN_ATTR __attribute__((noinline))
#else
#define ATT_FN_ATTR
#endif
ATT_FN_ATTR void attn_unit_256(const __global LayerTab* L, const __global ModelTab* mt, const int layer, const int u,
                               const int NS, const int past, const int jlo, const int jhi, const float posf,
                               const __global int* am, __local float* slm, const int att_base) {
    attn_unit(L, mt, layer, u, NS, past, jlo, jhi, posf, am, slm, att_base, 256, 256);
}
ATT_FN_ATTR void attn_unit_512(const __global LayerTab* L, const __global ModelTab* mt, const int layer, const int u,
                               const int NS, const int past, const int jlo, const int jhi, const float posf,
                               const __global int* am, __local float* slm, const int att_base) {
    attn_unit(L, mt, layer, u, NS, past, jlo, jhi, posf, am, slm, att_base, 512, 512);
}

// Warms L2 with what attn_unit of work-group `u` will read (issued before the
// grid barrier that precedes the attention phase).
inline void attn_prefetch(const __global LayerTab* L, const __global ModelTab* mt, const int u, const int NS,
                          const int jlo, const int jhi) {
    const int HK = L->HK, D = L->D;
    if (u >= HK * NS)
        return;
    const int kvh = u / NS;
    const int2 jr = split_keys(jlo, jhi, NS, u % NS);
    const int cap = mt->kv_cap;
    const __global uint* kc = (const __global uint*)(L->kc + (size_t)kvh * cap * D);
    const __global uint* vc = (const __global uint*)(L->vc + (size_t)kvh * cap * D);
    for (int j = jr.x + get_sub_group_id(); j < jr.y; j += NSG)
        for (int c = 0; c < D / 256; ++c) {
            intel_sub_group_block_prefetch_ui8(kc + (size_t)j * D / 2 + c * 128);
            intel_sub_group_block_prefetch_ui8(vc + (size_t)j * D / 2 + c * 128);
        }
    if (get_sub_group_id() == 0) {
        intel_sub_group_block_prefetch_ui8((const __global uint*)L->gq);
        intel_sub_group_block_prefetch_ui8((const __global uint*)L->gk);
        intel_sub_group_block_prefetch_ui8((const __global uint*)L->invf);
    }
}

// ---------------------------------------------------------------------------
// lm_head: u8 [V_, H] with a per-row scale / zero point, repacked as records of
// 16 rows x 64 bytes (dword j * 16 + lane = row lane, bytes 4j..4j+3). The
// activation is quantised in natural order (dword d of group g = elements
// 64g + 4d .. 4d + 3) as two int8 planes like the int4 GEMVs.
// ---------------------------------------------------------------------------
#define LM_REC_DW 256

inline void store_qn_l(QB q, float8 v, int b, int K, __local uint* x, int8 hi, int8 lo) {
    x[2 * b] = pack4(hi.s0, hi.s1, hi.s2, hi.s3);
    x[2 * b + 1] = pack4(hi.s4, hi.s5, hi.s6, hi.s7);
    x[K / 4 + 2 * b] = pack4(lo.s0, lo.s1, lo.s2, lo.s3);
    x[K / 4 + 2 * b + 1] = pack4(lo.s4, lo.s5, lo.s6, lo.s7);
    if ((get_sub_group_local_id() & 7) == 0) {
        x[K / 2 + (b >> 3)] = as_uint(q.a);
        x[K / 2 + K / QG + (b >> 3)] = as_uint(q.gsum);
    }
}

// Natural-order variant of quant_slm (same arithmetic as quant_block).
inline void quant_slm_nat(const __local float* src, const int K, __local uint* xq) {
    for (int c0 = get_sub_group_id() * SG; c0 < K / 8; c0 += NSG * SG) {
        const int b = c0 + get_sub_group_local_id();
        const bool ok = b < K / 8;
        const float8 v = ok ? vload8(b, src) : (float8)(0.0f);
        const QB q = quant_block(v, ok);
        const float8 t = v * (1.0f / q.a);
        const float8 hf = rint(t);
        const int8 hi = convert_int8(hf);
        const int8 lo = convert_int8(rint((t - hf) * XL_RES));
        if (ok)
            store_qn_l(q, v, b, K, xq, hi, lo);
    }
}

inline float lm_row_dot(const __global uint* rec, const __local uint* xq) {
    const __local uint* xh = xq;
    const __local uint* xl = xq + H_ / 4;
    const __local float* xa = (const __local float*)(xq + H_ / 2);
    float acc = 0.0f;
    __attribute__((opencl_unroll_hint(2))) for (int g = 0; g < H_ / QG; ++g) {
        const __global uint* p = rec + g * LM_REC_DW;
        const uint8 w0 = intel_sub_group_block_read8(p);
        const uint8 w1 = intel_sub_group_block_read8(p + 128);
        const uint xhv = intel_sub_group_block_read(xh + g * 16);
        const uint xlv = intel_sub_group_block_read(xl + g * 16);
        int ih = 0, il = 0;
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 8; ++j) {
            ih += dot_4x8packed_us_int(w0[j], sub_group_broadcast(xhv, j));
            il += dot_4x8packed_us_int(w0[j], sub_group_broadcast(xlv, j));
        }
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 8; ++j) {
            ih += dot_4x8packed_us_int(w1[j], sub_group_broadcast(xhv, 8 + j));
            il += dot_4x8packed_us_int(w1[j], sub_group_broadcast(xlv, 8 + j));
        }
        acc = fma(xa[g], fma((float)il, 1.0f / XL_RES, (float)ih), acc);
    }
    return acc;
}

inline float softcap(const __global ModelTab* mt, float y) {
    return mt->cap_div > 0.0f ? tanh(y / mt->cap_div) * mt->cap_mul : y;
}

// ---------------------------------------------------------------------------
// The decode megakernel.
// ---------------------------------------------------------------------------
#define SLM_XF 0                                   // float [H_]
#define SLM_XQ (H_)                                // uint  [XQ_WORDS(8192)]
#define SLM_XQ2 (SLM_XQ + XQ_WORDS(H_))            // second activation (phase E)
#define SLM_JOBS (SLM_XQ + XQ_WORDS(8192))         // Job[24]
#define SLM_TK (SLM_JOBS + 24 * 10)                // int[TOPK_] + float[TOPK_]
#define SLM_TMP (SLM_TK + 2 * TOPK_)               // float[NSG]
#define SLM_RED (SLM_TMP + NSG)                    // float[NSG * SG]
#define SLM_ATT_WORDS (NSG * ATT_QD + 4 * NSG + 512 + ATT_DMAX + 4 + 2 * ATT_DMAX + 16)
#define SLM_WORDS (SLM_RED + NSG * SG > SLM_ATT_WORDS ? SLM_RED + NSG * SG : SLM_ATT_WORDS)

REQD __kernel void mk_decode(const __global ModelTab* mt, const __global LayerTab* lt,
                             const __global float* embeds, const __global int* pos, const __global int* am,
                             __global float* logits, const int past, const int L, const int epoch, const int ns_s,
                             const int ns_f, const int att_base_s, const int att_base_f) {
    __local uint slm[SLM_WORDS];
    __local float* xf = (__local float*)(slm + SLM_XF);
    __local uint* xq = slm + SLM_XQ;
    __local Job* jobs = (__local Job*)(slm + SLM_JOBS);
    __local int* tk_i = (__local int*)(slm + SLM_TK);
    __local float* tk_w = (__local float*)(slm + SLM_TK + TOPK_);
    __local float* tmp = (__local float*)(slm + SLM_TMP);
    __local float* red = (__local float*)(slm + SLM_RED);

    const int wg = get_group_id(0);
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const int lid = get_local_id(0);
    const int gs = wg * NSG + sg;
    const int target = epoch * NWG;
    __global int* sync = mt->sync;
    __global int* err = mt->err;
    int bar = 0;
#define GRID_SYNC()                              \
    {                                            \
        grid_arrive(sync + bar);                 \
        PROF_MARK(mt, bar, 0)                    \
        grid_wait(sync + bar, target, err);      \
        PROF_MARK(mt, bar, 1)                    \
        ++bar;                                   \
    }
    PROF_MARK(mt, NL_ * NPH + 1, 0)

    float posf = (float)pos[0];
    // Key window of the current query (mask MegaKernel semantics, query = past).
    const int jhi = min(L - 1, past + (int)mt->causal_off - 1);
    const int jlo_s = max(0, past - (int)mt->window + 1);

    for (int l = 0; l < NL_; ++l) {
        const __global LayerTab* Ly = lt + l;
        const int R = Ly->R;

        // ---------------- A: residual, input RMS, q|k|v ----------------
        {
            float xv[HPT_];
            if (l == 0) {
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
                    xv[i] = (float)convert_half(embeds[lid + i * LWS]);
            } else {
                ffn_out_vals(lt + l - 1, mt, xv, tmp);
            }
            if (wg == 0) {
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
                    vstore_half(xv[i], lid + i * LWS, mt->xbuf);
#ifdef MK_DEBUG
                    vstore_half(xv[i], lid + i * LWS, (__global half*)mt->prof + (size_t)l * H_);
#endif
                }
            }
            rms_to_slm(xv, Ly->w_in, xf, tmp);
            barrier(CLK_LOCAL_MEM_FENCE);
            quant_slm(xf, H_, xq);
            if (lid == 0) {
                Job j = {Ly->qkv, mt->raw, H_ / QG, R / SG * (H_ / QG), SLM_XQ, H_, 0, 1.0f};
                jobs[0] = j;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            PROF_MARK(mt, bar, 2)
            run_jobs(jobs, 1, gs, NSGT, slm);
            attn_prefetch(Ly, mt, wg, Ly->sliding ? ns_s : ns_f, Ly->sliding ? jlo_s : 0, jhi);
        }
        GRID_SYNC();

        // ---------------- B: attention ----------------
        // Zero accumulators whose last reader has passed: gu, egu (layer l-1),
        // m, e (read in A).
        zero_grid(mt->gu, 2 * I_);
        zero_grid(mt->egu, TOPK_ * 2 * IE_);
        zero_grid(mt->m, H_);
        zero_grid(mt->e, H_);
        {
            const int NS = Ly->sliding ? ns_s : ns_f;
            const int jlo = Ly->sliding ? jlo_s : 0;
            if (Ly->pos_f16)
                posf = (float)convert_half_rte(posf);
            if (wg < Ly->HK * NS)
                {
#ifndef ATT_ONLY512
                if (Ly->D == 256)
                    attn_unit_256(Ly, mt, l, wg, NS, past, jlo, jhi, posf, am, xf, att_base_s);
#endif
#ifndef ATT_ONLY256
#ifndef ATT_ONLY512
                else
#endif
                    attn_unit_512(Ly, mt, l, wg, NS, past, jlo, jhi, posf, am, xf, att_base_f);
#endif
            }
            else {
                // DRAM is idle during attention: the other work-groups pull
                // (up to OPROJ_PF_MAX records of) o_proj into L2.
                const int busy = Ly->HK * NS * NSG;
                const int tot = min(OPROJ_PF_MAX, H_ / SG * (Ly->HQ * Ly->D / QG));
                const int r0 = slice_lo(tot, gs - busy, NSGT - busy), r1 = slice_lo(tot, gs - busy + 1, NSGT - busy);
                prefetch_recs(Ly->oproj + (size_t)r0 * REC_DW, r1 - r0);
            }
            posf = (float)pos[0];
        }
        GRID_SYNC();

        // ---------------- C: o_proj ----------------
        zero_grid(mt->raw, R);
        {
            const int KA = Ly->HQ * Ly->D;
#ifdef MK_DEBUG
            if (wg == 0)
                for (int i = lid; i < KA; i += LWS)
                    ((__global half*)mt->prof)[(size_t)NL_ * H_ + (size_t)l * 8192 + i] = mt->att[i];
#endif
            const __global half* att = mt->att;
#define LOAD_ATT(b) vload_half8((b), att)
            QUANT_GLOBAL_WITH(LOAD_ATT, KA, xq)
#undef LOAD_ATT
            if (lid == 0) {
                Job j = {Ly->oproj, mt->o, KA / QG, H_ / SG * (KA / QG), SLM_XQ, KA, 0, 1.0f};
                jobs[0] = j;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            PROF_MARK(mt, bar, 2)
            run_jobs(jobs, 1, gs, NSGT, slm);
            // Next: the dense gate projection of D.
            if (wg < ROUTER_WG0) {
                const int per = I_ / SG * (H_ / QG);
                const int r0 = slice_lo(2 * per, gs, ROUTER_WG0 * NSG);
                const int r1 = slice_lo(2 * per, gs + 1, ROUTER_WG0 * NSG);
                const __global uint* m0 = r0 < per ? Ly->gate + (size_t)r0 * REC_DW : Ly->up + (size_t)(r0 - per) * REC_DW;
                prefetch_recs(m0, min(PF_RECS, min(r1, r0 < per ? per : 2 * per) - r0));
            }
        }
        GRID_SYNC();

        // ---------------- D: h1, router, dense gate/up, MoE gate/up ----------------
        // Every work-group first adds its slice of the router logits, then
        // streams its dense gate/up share; by then all router slices are in,
        // so the MoE gate/up projections follow inside the same phase.
        {
            const __global long* o = mt->o;
            float hv[HPT_];
            float ss = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
                hv[i] = fx2f(o[lid + i * LWS]);
                ss += hv[i] * hv[i];
            }
            const float inv_o = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
            ss = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
                const int k = lid + i * LWS;
                hv[i] = vload_half(k, mt->xbuf) + hv[i] * inv_o * vload_half(k, Ly->w_pa);
                ss += hv[i] * hv[i];
            }
            if (wg == 0) {
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
                    mt->h1[lid + i * LWS] = hv[i];
            }
            const float inv1 = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
                xf[lid + i * LWS] = hv[i] * inv1 * vload_half(lid + i * LWS, Ly->w_pf2);
            barrier(CLK_LOCAL_MEM_FENCE);
            quant_slm(xf, H_, slm + SLM_XQ2);
            barrier(CLK_LOCAL_MEM_FENCE);
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
                xf[lid + i * LWS] = hv[i] * inv1 * vload_half(lid + i * LWS, Ly->w_pf);
            barrier(CLK_LOCAL_MEM_FENCE);
            quant_slm(xf, H_, xq);
            barrier(CLK_LOCAL_MEM_FENCE);
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
                xf[lid + i * LWS] = hv[i] * inv1 * vload_half(lid + i * LWS, Ly->w_r);
            if (lid < 2) {
                Job j = {lid ? Ly->up : Ly->gate, mt->gu + lid * I_, H_ / QG, I_ / SG * (H_ / QG), SLM_XQ, H_, 0,
                         1.0f};
                jobs[lid] = j;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            PROF_MARK(mt, bar, 2)
            // Router: f16 [E_, H_], items of 16 rows x 16 columns (row per lane).
            {
                const int NIT = (E_ / SG) * (H_ / SG);
                int i = slice_lo(NIT, gs, NSGT);
                const int i1 = slice_lo(NIT, gs + 1, NSGT);
                while (i < i1) {
                    const int rb = i / (H_ / SG);
                    const int e = min(i1, (rb + 1) * (H_ / SG));
                    const __global half* row = Ly->rw + (size_t)(rb * SG + ln) * H_;
                    float acc = 0.0f;
                    for (; i < e; ++i) {
                        const int kc = i - rb * (H_ / SG);
                        const float16 w = vload_half16(kc, row);
                        const float x = xf[kc * SG + ln];
                        __attribute__((opencl_unroll_hint)) for (int j = 0; j < SG; ++j)
                            acc = fma(w[j], sub_group_broadcast(x, j), acc);
                    }
                    atomic_addfx(mt->lg + rb * SG + ln, acc);
                }
            }
            __global int* rdone = sync + NL_ * NPH + 1 + l;
            grid_arrive(rdone);
            run_jobs(jobs, 2, gs, NSGT, slm);
            PROF_MARK(mt, bar, 3)
            grid_wait(rdone, target, err);
            if (sg == 0)
                topk_local(mt->lg, tk_i, tk_w);
            barrier(CLK_LOCAL_MEM_FENCE);
            PROF_MARK(mt, bar, 4)
            if (lid < 2 * TOPK_) {
                const int t = lid % TOPK_, up = lid / TOPK_;
                const size_t eoff = (size_t)tk_i[t] * (IE_ / SG) * (H_ / QG) * REC_DW;
                Job j = {(up ? Ly->eup : Ly->egate) + eoff, mt->egu + (t * 2 + up) * IE_, H_ / QG,
                         IE_ / SG * (H_ / QG), SLM_XQ2, H_, 0, 1.0f};
                jobs[up * TOPK_ + t] = j;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            run_jobs(jobs, 2 * TOPK_, gs, NSGT, slm);
            // Next: E's down projections (experts known now).
            {
                const int per = H_ / SG * (IE_ / QG);
                const int tot = TOPK_ * per + H_ / SG * (I_ / QG);
                const int r0 = slice_lo(tot, gs, NSGT), r1 = slice_lo(tot, gs + 1, NSGT);
                if (r0 < TOPK_ * per) {
                    const int t = r0 / per;
                    prefetch_recs(Ly->edown + ((size_t)tk_i[t] * per + (r0 - t * per)) * REC_DW,
                                  min(PF_RECS, min(r1, (t + 1) * per) - r0));
                } else {
                    prefetch_recs(Ly->down + (size_t)(r0 - TOPK_ * per) * REC_DW, min(PF_RECS, r1 - r0));
                }
            }
        }
        GRID_SYNC();

        // ---------------- E0: gelu(gate) * up, quantised once ----------------
        // Two 64-element groups per sub-group; the expert vector (TOPK_ * IE_)
        // and the dense one (I_) are quantised into global buffers.
        {
            const int NGM = TOPK_ * IE_ / QG, NGD = I_ / QG;
            const int gi = gs * 2 + ln / 8;
            if (gs * 2 < NGM + NGD) {
                const bool ok = gi < NGM + NGD;
                const bool moe = gi < NGM;
                const int gl = moe ? gi : gi - NGM;             // group within its vector
                const int b = gl * 8 + (ln & 7);                // 8-block within its vector
                float8 v = (float8)(0.0f);
                if (ok) {
                    const __global long* gp = moe ? mt->egu + (b / (IE_ / 8)) * 2 * IE_ : mt->gu;
                    const int c = moe ? b % (IE_ / 8) : b;
                    v = gelu_tanh8(fx2f8(vload8(c, gp))) * fx2f8(vload8(c, gp + (moe ? IE_ : I_)));
                }
                const QB q = quant_block(v, ok);
                if (ok)
                    store_qb_g(q, b, moe ? TOPK_ * IE_ : I_, moe ? mt->xqm : mt->xqd);
            }
        }
        GRID_SYNC();

        // ---------------- E: MoE down, dense down ----------------
        zero_grid(mt->o, H_);
        zero_grid(mt->lg, E_);
        {
            for (int i = lid; i < XQ_WORDS(TOPK_ * IE_); i += LWS)
                xq[i] = mt->xqm[i];
            for (int i = lid; i < XQ_WORDS(I_); i += LWS)
                slm[SLM_XQ2 + i] = mt->xqd[i];
            if (lid < TOPK_) {
                const size_t eoff = (size_t)tk_i[lid] * (H_ / SG) * (IE_ / QG) * REC_DW;
                Job j = {Ly->edown + eoff, mt->e, IE_ / QG, H_ / SG * (IE_ / QG), SLM_XQ, TOPK_ * IE_,
                         lid * (IE_ / QG), tk_w[lid]};
                jobs[lid] = j;
            } else if (lid == TOPK_) {
                Job j = {Ly->down, mt->m, I_ / QG, H_ / SG * (I_ / QG), SLM_XQ2, I_, 0, 1.0f};
                jobs[TOPK_] = j;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            PROF_MARK(mt, bar, 2)
            run_jobs(jobs, TOPK_ + 1, gs, NSGT, slm);
            // Next: q|k|v of the next layer.
            if (l + 1 < NL_)
                prefetch_single(lt[l + 1].qkv, lt[l + 1].R / SG * (H_ / QG), gs, NSGT);
        }
        GRID_SYNC();
    }

    // ---------------- G: final norm, lm_head, soft-cap ----------------
    zero_grid(mt->egu, TOPK_ * 2 * IE_);
    {
        float xv[HPT_];
        ffn_out_vals(lt + NL_ - 1, mt, xv, tmp);
        rms_to_slm(xv, mt->w_final, xf, tmp);
        barrier(CLK_LOCAL_MEM_FENCE);
        quant_slm_nat(xf, H_, xq);
        barrier(CLK_LOCAL_MEM_FENCE);
        float sx = 0.0f;
        const __local float* xs = (const __local float*)(xq + H_ / 2 + H_ / QG);
        for (int g = 0; g < H_ / QG; ++g)
            sx += xs[g];
        PROF_MARK(mt, bar, 2)
        const int nrb = V_ / SG;
        const int rb0 = slice_lo(nrb, gs, NSGT), rb1 = slice_lo(nrb, gs + 1, NSGT);
        for (int rb = rb0; rb < rb1; ++rb) {
            const float acc = lm_row_dot(mt->lm + (size_t)rb * (H_ / QG) * LM_REC_DW, xq);
            const int row = rb * SG + ln;
            const float y = (float)mt->lm_s[row] * (acc - (float)mt->lm_z[row] * sx);
            logits[row] = softcap(mt, y);
        }
    }
    PROF_MARK(mt, NL_ * NPH + 1, 1)
#undef GRID_SYNC
}

// ===========================================================================
// Prompt (multi-token) helpers of the whole-model path.
// ===========================================================================
__kernel void mk_cvt_embeds(const __global float* in, __global half* out, const int n) {
    const int i = get_global_id(0);
    if (i < n)
        out[i] = convert_half(in[i]);
}

// K / V heads of attn_post's head-major output [HQ + 2 HK][T][D] -> cache rows past..past+T-1.
__kernel void mk_kv_append(const __global half* hm, const int T, const int HQ, const int HK, const int D,
                           __global half* kc, __global half* vc, const int cap, const int past) {
    const int d = get_global_id(0);
    const int t = get_global_id(1);
    const int h = get_global_id(2);
    if (d >= D || t >= T)
        return;
    const int kv = h % HK;
    const __global half* src = hm + ((size_t)(HQ + h) * T + t) * D;
    __global half* dst = (h < HK ? kc : vc) + ((size_t)kv * cap + past + t) * D;
    dst[d] = src[d];
}

// First and last visible key of every query row of a [T, L] f16 mask.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_mask_range(const __global half* mask, const int T, const int L, __global int* lo, __global int* hi) {
    const int t = get_global_id(1);
    if (t >= T)
        return;
    const __global half* row = mask + (size_t)t * L;
    int a = L, b = -1;
    for (int j = get_sub_group_local_id(); j < L; j += SG) {
        if ((float)row[j] > -1e4f) {
            a = min(a, j);
            b = max(b, j);
        }
    }
    a = sub_group_reduce_min(a);
    b = sub_group_reduce_max(b);
    if (get_sub_group_local_id() == 0) {
        lo[t] = a;
        hi[t] = b;
    }
}

// Prompt attention: one sub-group per (head, QB = 512 / D consecutive queries),
// online softmax over the visible keys of the cache with the explicit mask.
// q comes from attn_post's head-major output, O is [T][HQ * D]. D is a literal
// at both call sites.
inline void pf_attn_body(const __global half* hm, const __global half* kcache, const __global half* vcache,
                         const __global half* mask, const __global int* mlo, const __global int* mhi, const int T,
                         const int L, const int HQ, const int HK, const int cap, const float scale,
                         __global half* O, const int D) {
    const int QB = ATT_QD / D;
    const int nj = D / SG;
    const int nqb = (T + QB - 1) / QB;
    const int w = get_group_id(0) * (get_local_size(0) / SG) + get_sub_group_id();
    if (w >= HQ * nqb)
        return;
    const int h = w / nqb, t0 = (w % nqb) * QB;
    const int kvh = h / (HQ / HK);
    const int ln = get_sub_group_local_id();
    float qv[ATT_QD / SG], acc[ATT_QD / SG];
    float mx[2], ls[2];
    int jl = L, jh = -1;
    __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
        const int t = min(t0 + q, T - 1);
        jl = min(jl, mlo[t]);
        jh = max(jh, mhi[t]);
        mx[q] = -INFINITY;
        ls[q] = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < nj; ++j) {
            qv[q * nj + j] = (float)hm[((size_t)h * T + t) * D + j * SG + ln] * scale;
            acc[q * nj + j] = 0.0f;
        }
    }
    const __global half* kc = kcache + (size_t)kvh * cap * D;
    const __global half* vc = vcache + (size_t)kvh * cap * D;
    // Keys go in chunks of SG: every lane forms the per-lane partial dots of
    // all SG keys, a butterfly transpose-reduce leaves key `ln`'s score on lane
    // ln (15 shuffles instead of SG full reductions), and the online softmax
    // is updated once per chunk instead of once per key.
    for (int j0 = jl; j0 <= jh; j0 += SG) {
        const int jk = j0 + ln;
        float madd[2];
        bool any = false;
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
            const int t = t0 + q;
            madd[q] = (t < T && jk <= jh) ? (float)mask[(size_t)t * L + jk] : -INFINITY;
            any |= madd[q] > -1e4f;
        }
        if (!sub_group_any(any))
            continue;
        float v[2][SG];
        __attribute__((opencl_unroll_hint)) for (int k = 0; k < SG; ++k) {
            const __global ushort* kp = (const __global ushort*)(kc + (size_t)min(j0 + k, jh) * D);
            float s[2] = {0.0f, 0.0f};
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c) {
                const half8 k8 = as_half8(intel_sub_group_block_read_us8(kp + c * 128));
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                    __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q)
                        s[q] = fma(qv[q * nj + c * 8 + i], (float)k8[i], s[q]);
            }
            __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q)
                v[q][k] = s[q];
        }
        float sc[2];
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
            __attribute__((opencl_unroll_hint)) for (int hb = SG / 2; hb >= 1; hb >>= 1) {
                const bool up = (ln & hb) != 0;
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < hb; ++i) {
                    const float send = up ? v[q][i] : v[q][i + hb];
                    const float keep = up ? v[q][i + hb] : v[q][i];
                    v[q][i] = keep + intel_sub_group_shuffle_xor(send, (uint)hb);
                }
            }
            sc[q] = v[q][0] + madd[q];
        }
        float pr[2];
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
            const bool vis = madd[q] > -1e4f;
            const float mn = fmax(mx[q], sub_group_reduce_max(vis ? sc[q] : -INFINITY));
            const float cr = mx[q] == -INFINITY ? 0.0f : exp(mx[q] - mn);
            pr[q] = vis ? exp(sc[q] - mn) : 0.0f;
            ls[q] = fma(ls[q], cr, sub_group_reduce_add(pr[q]));
            mx[q] = mn;
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < nj; ++j)
                acc[q * nj + j] *= cr;
        }
        const int nk = min(SG, jh - j0 + 1);
        for (int k = 0; k < nk; ++k) {
            float pk[2];
            bool nz = false;
            __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
                pk[q] = sub_group_broadcast(pr[q], k);
                nz |= pk[q] != 0.0f;
            }
            if (!nz)
                continue;
            const __global ushort* vp = (const __global ushort*)(vc + (size_t)(j0 + k) * D);
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c) {
                const half8 v8 = as_half8(intel_sub_group_block_read_us8(vp + c * 128));
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                    __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
                        const int kk = q * nj + c * 8 + i;
                        acc[kk] = fma(pk[q], (float)v8[i], acc[kk]);
                    }
            }
        }
    }
    __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
        const int t = t0 + q;
        if (t < T) {
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < nj; ++j)
                O[(size_t)t * HQ * D + h * D + j * SG + ln] = convert_half(acc[q * nj + j] / ls[q]);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_pf_attn(const __global half* hm, const __global half* kcache, const __global half* vcache,
                         const __global half* mask, const __global int* mlo, const __global int* mhi, const int T,
                         const int L, const int HQ, const int HK, const int D, const int cap, const float scale,
                         __global half* O) {
    if (D == 256)
        pf_attn_body(hm, kcache, vcache, mask, mlo, mhi, T, L, HQ, HK, cap, scale, O, 256);
    else
        pf_attn_body(hm, kcache, vcache, mask, mlo, mhi, T, L, HQ, HK, cap, scale, O, 512);
}

// XMX flash attention for the prompt path. One sub-group = one head, 8
// consecutive queries and a 256-wide slice of the output (D = 512 heads use
// two sub-groups that both compute the scores). Per 16-key chunk:
//   S[8x16] = Q[8xD] K^T   DPAS; Q tiles are 2D block reads of hm (rows past T
//                          read 0), K^T comes from a transposed 32-bit read
//   online softmax per query row (one max reduction per row and chunk)
//   O[8x256] += P V        DPAS; V tiles via the VNNI-transforming read
// Rows past jh are outside the read height and come back as zeros.
#define FA_SGS 8
__attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(SG * FA_SGS, 1, 1)))
__kernel void mk_pf_fa(const __global half* hm, const __global half* kcache, const __global half* vcache,
                       const __global half* mask, const __global int* mlo, const __global int* mhi, const int T,
                       const int L, const int HQ, const int HK, const int D, const int cap, const float scale,
                       __global half* O) {
    const int ndh = D / 256;
    const int nqb = (T + 7) / 8;
    const int w = get_group_id(0) * FA_SGS + get_sub_group_id();
    if (w >= HQ * nqb * ndh)
        return;
    const int dh = w % ndh, qb = (w / ndh) % nqb, h = w / (ndh * nqb);
    const int t0 = qb * 8;
    const int kvh = h / (HQ / HK);
    const int ln = get_sub_group_local_id();
    int jl = L, jh = -1;
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
        const int t = min(t0 + m, T - 1);
        jl = min(jl, mlo[t]);
        jh = max(jh, mhi[t]);
    }
    const __global half* qp = hm + (size_t)h * T * D;
    const __global half* kc = kcache + (size_t)kvh * cap * D;
    const __global half* vc = vcache + (size_t)kvh * cap * D;
    const int W = D * 2, HT = jh + 1;
    float8 acc[16];
    __attribute__((opencl_unroll_hint)) for (int c = 0; c < 16; ++c)
        acc[c] = (float8)(0.0f);
    float8 lsum = (float8)(0.0f);
    float mx[8];
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m)
        mx[m] = -INFINITY;
    for (int j0 = jl; j0 <= jh; j0 += 16) {
        const int j = j0 + ln;
        float8 madd;
        int vany = 0;
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const int t = t0 + m;
            madd[m] = (t < T && j <= jh) ? (float)mask[(size_t)t * L + j] : -INFINITY;
            vany |= madd[m] > -1e4f;
        }
        if (!sub_group_any(vany))
            continue;
        float8 s = (float8)(0.0f);
        for (int c = 0; c < D / 16; ++c) {
            ushort8 a;
            uint8 b;
            intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)qp, W, T, W, (int2)(c * 16, t0),
                                                      (__private ushort*)&a);
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)kc, W, HT, W, (int2)(c * 8, j0),
                                                                (__private uint*)&b);
            s = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), as_int8(b), s);
        }
        float8 p, cr;
        int rescale = 0;
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const bool vis = madd[m] > -1e4f;
            const float sc = fma(s[m], scale, madd[m]);
            const float mn = fmax(mx[m], sub_group_reduce_max(vis ? sc : -INFINITY));
            cr[m] = mx[m] == -INFINITY ? 0.0f : exp(mx[m] - mn);
            rescale |= cr[m] != 1.0f;
            p[m] = vis ? exp(sc - mn) : 0.0f;
            mx[m] = mn;
        }
        lsum = fma(lsum, cr, p);
        if (rescale)
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < 16; ++c)
                acc[c] *= cr;
        const short8 pa = as_short8(convert_half8(p));
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < 16; ++c) {
            uint8 b;
            intel_sub_group_2d_block_read_transform_16b_16r16x1c((__global void*)vc, W, HT, W,
                                                                 (int2)(dh * 256 + c * 16, j0), (__private uint*)&b);
            acc[c] = intel_sub_group_f16_f16_matrix_mad_k16(pa, as_int8(b), acc[c]);
        }
    }
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
        const int t = t0 + m;
        if (t >= T)
            break;
        const float l = sub_group_reduce_add(lsum[m]);
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        __global ushort* orow = (__global ushort*)(O + (size_t)t * HQ * D + h * D + dh * 256);
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < 16; ++c)
            intel_sub_group_block_write_us(orow + c * 16, as_ushort(convert_half(acc[c][m] * inv)));
    }
}

// Final norm + lm_head + soft-cap for `rows` tokens of a [rows, H] f16 residual.
// Work-group: LM_TB tokens x 16 sub-groups, each sub-group LM_RB row blocks.
#define LM_TB 4
#define LM_RB 4
REQD __kernel void mk_lm_head_rows(const __global ModelTab* mt, const __global half* x, const int rows,
                                   __global float* logits) {
    __local uint xq[LM_TB][XQ_WORDS(H_)];
    __local float xf[H_];
    __local float tmp[NSG];
    __local float sxs[LM_TB];
    // Token groups vary fastest across work-groups, so the groups in flight
    // share one weight slice in L2 instead of each streaming all of lm_head.
    const int t0 = get_group_id(0) * LM_TB;
    const int lid = get_local_id(0);
    for (int q = 0; q < LM_TB; ++q) {
        const int t = min(t0 + q, rows - 1);
        float v[HPT_];
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
            v[i] = vload_half(lid + i * LWS, x + (size_t)t * H_);
        rms_to_slm(v, mt->w_final, xf, tmp);
        barrier(CLK_LOCAL_MEM_FENCE);
        quant_slm_nat(xf, H_, xq[q]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
            float sx = 0.0f;
            const __local float* xs = (const __local float*)(xq[q] + H_ / 2 + H_ / QG);
            for (int g = 0; g < H_ / QG; ++g)
                sx += xs[g];
            sxs[q] = sx;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int ln = get_sub_group_local_id();
    const int rb0 = (get_group_id(1) * NSG + get_sub_group_id()) * LM_RB;
    for (int rb = rb0; rb < min(rb0 + LM_RB, V_ / SG); ++rb) {
        const __global uint* rec = mt->lm + (size_t)rb * (H_ / QG) * LM_REC_DW;
        float acc[LM_TB];
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < LM_TB; ++q)
            acc[q] = 0.0f;
        for (int g = 0; g < H_ / QG; ++g) {
            const __global uint* p = rec + g * LM_REC_DW;
            const uint8 w0 = intel_sub_group_block_read8(p);
            const uint8 w1 = intel_sub_group_block_read8(p + 128);
            __attribute__((opencl_unroll_hint)) for (int q = 0; q < LM_TB; ++q) {
                const uint xhv = intel_sub_group_block_read(xq[q] + g * 16);
                const uint xlv = intel_sub_group_block_read(xq[q] + H_ / 4 + g * 16);
                int ih = 0, il = 0;
                __attribute__((opencl_unroll_hint)) for (int j = 0; j < 8; ++j) {
                    ih += dot_4x8packed_us_int(w0[j], sub_group_broadcast(xhv, j));
                    il += dot_4x8packed_us_int(w0[j], sub_group_broadcast(xlv, j));
                    ih += dot_4x8packed_us_int(w1[j], sub_group_broadcast(xhv, 8 + j));
                    il += dot_4x8packed_us_int(w1[j], sub_group_broadcast(xlv, 8 + j));
                }
                const float a = ((const __local float*)(xq[q] + H_ / 2))[g];
                acc[q] = fma(a, fma((float)il, 1.0f / XL_RES, (float)ih), acc[q]);
            }
        }
        const int row = rb * SG + ln;
        const float s = (float)mt->lm_s[row], z = (float)mt->lm_z[row];
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < LM_TB; ++q) {
            const int t = t0 + q;
            if (t < rows)
                logits[(size_t)t * V_ + row] = softcap(mt, s * (acc[q] - z * sxs[q]));
        }
    }
}
)CLC";

// Host-side launch geometry of mk_lm_head_rows (must match LM_TB / LM_RB).
constexpr int kLmTb = 4;
constexpr int kLmRb = 4;
