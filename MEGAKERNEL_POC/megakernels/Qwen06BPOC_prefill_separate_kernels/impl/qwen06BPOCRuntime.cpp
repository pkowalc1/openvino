#include "qwen06BPOCRuntime.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mk {
namespace {
template <typename... Args>
[[noreturn]] void throw_error(Args&&... args) {
    std::ostringstream message;
    (message << ... << std::forward<Args>(args));
    throw std::runtime_error(message.str());
}

template <typename... Args>
void assert_or_throw(bool condition, Args&&... args) {
    if (!condition)
        throw_error(std::forward<Args>(args)...);
}
}  // namespace

// ---------------------------------------------------------------------------
// Kernel source — embedded attempt4 kernels adapted for plugin tensor layouts
// ---------------------------------------------------------------------------
static const char* kKernelSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16              : enable
#pragma OPENCL EXTENSION cl_intel_subgroups       : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define H     1024
#define QDIM  2048
#define KVDIM 1024
#define HD    128
#define NH    16
#define KVH   8
#define HHD   64
#define GQA   2
#define IM    3072
#define EPS   1e-6f
#define SG    32
#define RPS   2
#define NUM_L 28

#define THREADS 512
#define TOTAL_WARPS (THREADS/SG)

#include "taskSystem/shared/taskDesc.h"
#include "common/semaphore.hcl"

// ---------------------------------------------------------------------------
// Compute RMS once per work-group; all GEMV subgroups consume the same value.
inline float wg_rms(const __global half* h, __local char* slm) {
    uint lid = get_local_id(0), lane = get_sub_group_local_id(), sgl = get_sub_group_id();
    float2 v = convert_float2(vload2(0, h + lid * 2));
    float ss = sub_group_reduce_add(dot(v, v));
    __local float* partial = (__local float*)slm;
    if (lane == 0)
        partial[sgl] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgl == 0) {
        ss = lane < get_num_sub_groups() ? partial[lane] : 0.0f;
        ss = sub_group_reduce_add(ss);
        if (lane == 0)
            partial[0] = rsqrt(ss / H + EPS);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return partial[0];
}

inline void wg_rms2(const __global half* h, const __global half* wn,
                    __global half* out, __local char* slm) {
    uint lid = get_local_id(0), lane = get_sub_group_local_id(), sgl = get_sub_group_id();
    float2 v = convert_float2(vload2(0, h + lid * 2));
    float ss = sub_group_reduce_add(dot(v, v));
    __local float* partial = (__local float*)slm;
    if (lane == 0)
        partial[sgl] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgl == 0) {
        ss = lane < get_num_sub_groups() ? partial[lane] : 0.0f;
        ss = sub_group_reduce_add(ss);
        if (lane == 0)
            partial[0] = rsqrt(ss / H + EPS);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 norm = convert_float2(vload2(0, wn + lid * 2));
    vstore2(convert_half2(v * partial[0] * norm), 0, out + lid * 2);
}

// GEMV with fused RMS + block reads (SIMD16 message per 256-element strip)
inline void sg_gemv_rms(const __global half* h, const __global half* wn, float rms,
                        const __global half* w, uint base, uint lane, float* out) {
    float acc[RPS];
    for (int r = 0; r < RPS; r++) acc[r] = 0;
    for (uint blk = 0; blk < H; blk += SG * 16) {
        const __global ushort* hp = (const __global ushort*)(h + blk);
        float8 hlo = convert_float8(as_half8(intel_sub_group_block_read_us8(hp)));
        float8 hhi = convert_float8(as_half8(intel_sub_group_block_read_us8(hp + SG * 8)));
        const __global ushort* np = (const __global ushort*)(wn + blk);
        float8 xlo = hlo*rms*convert_float8(as_half8(intel_sub_group_block_read_us8(np)));
        float8 xhi = hhi*rms*convert_float8(as_half8(intel_sub_group_block_read_us8(np + SG*8)));
        for (int r = 0; r < RPS; r++) {
            const __global ushort* wp = (const __global ushort*)(w + (ulong)(base+r)*H + blk);
            float8 ylo = convert_float8(as_half8(intel_sub_group_block_read_us8(wp)));
            float8 yhi = convert_float8(as_half8(intel_sub_group_block_read_us8(wp + SG*8)));
            float8 p = xlo*ylo + xhi*yhi;
            acc[r] += p.s0+p.s1+p.s2+p.s3+p.s4+p.s5+p.s6+p.s7;
        }
    }
    for (int r = 0; r < RPS; r++) out[r] = sub_group_reduce_add(acc[r]);
}

#define NPL (HD/SG)   // head-dim elements handled per subgroup lane (attention)

// Plain subgroup GEMV over an fp16 activation (no fused RMS, no split-K):
// each subgroup fully reduces RPS output rows. Used for o-proj / down-proj.
inline void sg_gemv_f16(const __global half* a, const __global half* w, uint IN,
                        uint base, uint lane, float* out) {
    float acc[RPS];
    for (int r=0; r<RPS; r++) acc[r]=0;
    for (uint blk=0; blk<IN; blk+=SG*16) {
        const __global ushort* ap=(const __global ushort*)(a+blk);
        float8 xlo=convert_float8(as_half8(intel_sub_group_block_read_us8(ap)));
        float8 xhi=convert_float8(as_half8(intel_sub_group_block_read_us8(ap+SG*8)));
        for (int r=0; r<RPS; r++) {
            const __global ushort* wp=(const __global ushort*)(w+(ulong)(base+r)*IN+blk);
            float8 ylo=convert_float8(as_half8(intel_sub_group_block_read_us8(wp)));
            float8 yhi=convert_float8(as_half8(intel_sub_group_block_read_us8(wp+SG*8)));
            float8 p=xlo*ylo+xhi*yhi;
            acc[r]+=p.s0+p.s1+p.s2+p.s3+p.s4+p.s5+p.s6+p.s7;
        }
    }
    for (int r=0; r<RPS; r++) out[r]=sub_group_reduce_add(acc[r]);
}

// ===========================================================================
// MEGAKERNEL TASKS: the 28-layer decoder expressed as task-system tasks.
// The persistent grid_barrier monokernel is replaced by a pool of task workers.
// Each layer stage is decomposed into per-workgroup tiles (tasks); a stage's
// tasks all wait (via a global atomic counter) for the previous stage to finish,
// exactly replicating the grid-barrier ordering, and signal their own counter on
// completion. The GEMV / RMSNorm / RoPE / flash-attention math is unchanged.
// ===========================================================================
#define SGN 16                                 // sub-groups per work-group (LWS 256 / SG 16)
#define TF  1                                  // GEMV tile coarsening (RPS-groups per lane per task)
#define NT_AQ (QDIM/(RPS*SGN*TF))              // Stage AQ (Q) tile count       = 64
#define NT_AK (KVDIM/(RPS*SGN*TF))             // Stage AK (K) tile count       = 32
#define NT_AV (KVDIM/(RPS*SGN*TF))             // Stage AV (V) tile count       = 32
#define NT_A (NT_AQ+NT_AK+NT_AV)               // Stage A total tile count      = 128
#define NT_BC (NH+KVH)                          // Stage BC (attn) task count    = 24
#define NT_D  (H/(RPS*SGN*TF))                  // Stage D (o-proj) tile count   = 32
#define NT_E  (IM/(RPS*SGN*TF))                 // Stage E (gate/up) tile count  = 96
#define NT_F  (H/(RPS*SGN*TF))                  // Stage F (down) tile count     = 32
#define EMBED_IDX (NUM_L*5)                     // sync slot for the embedding stage
#define SLM_BYTES ((2*SGN + SGN*NPL*SG)*4)      // flash-decoding partials (lsm_m/lsm_l/lsm_a)

// Per-token context shared by every task: all base pointers plus the scalars
// (pos, cache stride, token offset) that vary per launch. Lives in USM device
// memory; each task carries a pointer to it in its payload.
typedef struct MonoCtx {
    __global const half*  hs;
    __global half*        h;
    __global float*       out;
    __global const half*  wn; __global const half* pn;
    __global const half*  qw; __global const half* kw; __global const half* vw;
    __global const half*  ow; __global const half* gw; __global const half* uw;
    __global const half*  dw;
    __global const half*  qn; __global const half* kn; __global const half* rf;
    __global half*        qb; __global half* kb; __global half* vb;
    __global half*        xn; __global half* gbuf;
    __global half*        nbuf;
    __global half*        kc; __global half* vc;
    __global int*         sync;
    __global long*        past_pos;
    int  step; uint CS; uint tok_off;
} MonoCtx;

typedef struct MkTask {
    __global const MonoCtx* ctx;
    int layer;
    int tile;
} MkTask;

// Stage 0: vectorized copy into the fp16 residual stream (single task).
inline void mk_embed(const MkTask t, __local char* slm) {
    __global const MonoCtx* c = t.ctx;
    __global const half* hs = c->hs + c->tok_off;
    __global half*       h  = c->h  + c->tok_off;
    uint i = get_local_id(0) * 2;
    vstore2(vload2(0, hs + i), 0, h + i);
    SignalSemaphore_block(0, (volatile __global atomic_int*)(c->sync + EMBED_IDX));
}

// Materialize the weighted, normalized activation before Stage A.
inline void mk_normA(const MkTask t, __local char* slm) {
    __global const MonoCtx* c = t.ctx;
    int layer = t.layer;
    int dep    = (layer == 0) ? EMBED_IDX : ((layer-1)*5 + 4);
    int depcnt = (layer == 0) ? 1         : NT_F;
    WaitForSemaphore_block(0, (volatile __global atomic_int*)(c->sync + dep), depcnt);
    wg_rms2(c->h + c->tok_off, c->wn + layer*H, c->nbuf, slm);
    SignalSemaphore_block(0, (volatile __global atomic_int*)(c->sync + layer*5 + 0));
}

// #define GemvBlock_MATRIX_ROWS 2048
// #define GemvBlock_MATRIX_COLUMNS 1024
// #define GemvBlock_BLOCK_TILE_ROWS 32
// #define GemvBlock_PHASE_TILE_ROWS 4
// #define GemvBlock_COMPUTE_WARPS 4
// #define GemvBlock_SUFFIX _2048x1024
// #include "gemvOpt/gemvBlock.hcl"

inline void mk_stageAQ(const MkTask t, __local char* slm) {
    __global const MonoCtx* c = t.ctx;
    int layer = t.layer, tile = t.tile;
    const uint wn_off=layer*H, qw_off=layer*QDIM*H;
    volatile __global atomic_int* sem = (volatile __global atomic_int*)(c->sync + layer*5 + 0);
    int dep = (layer == 0) ? EMBED_IDX : ((layer-1)*5 + 4);
    int depcnt = (layer == 0) ? 1 : NT_F;

    uint l = get_sub_group_local_id(), sgl = get_sub_group_id();
    WaitForSemaphore_block(0, (volatile __global atomic_int*)(c->sync + dep), depcnt);
    __global half* h = c->h + c->tok_off;
    float rms = wg_rms(h, slm);
    for (int g = 0; g < TF; g++) {
        uint gi = (uint)tile*(SGN*TF) + (uint)g*SGN + sgl;
        uint n = gi*RPS;
        float o[RPS];
        sg_gemv_rms(h, c->wn+wn_off, rms, c->qw+qw_off, n, l, o);
        if (l==0) vstore2(convert_half2((float2)(o[0], o[1])), 0, c->qb+n);
    }

    // const __global half* vector = c->nbuf;
    // const __global half* matrix = c->qw + qw_off;
    // __global half* output = c->qb;
    // GemvBlock_2048x1024(tile, matrix, vector, output,
    //                 slm,
    //                 sem,
    //                 1);

    SignalSemaphore_block(0, sem);
}

// #define GemvBlock_MATRIX_ROWS 1024
// #define GemvBlock_MATRIX_COLUMNS 1024
// #define GemvBlock_BLOCK_TILE_ROWS 32
// #define GemvBlock_PHASE_TILE_ROWS 8
// #define GemvBlock_COMPUTE_WARPS 4
// #define GemvBlock_SUFFIX _1024x1024
// #include "gemvOpt/gemvBlock.hcl"

inline void mk_stageAK(const MkTask t, __local char* slm) {
    __global const MonoCtx* c = t.ctx;
    int layer = t.layer, tile = t.tile;
    uint wn_off=layer*H, kw_off=layer*KVDIM*H;
    volatile __global atomic_int* sem = (volatile __global atomic_int*)(c->sync + layer*5 + 0);
    int dep = (layer == 0) ? EMBED_IDX : ((layer-1)*5 + 4);
    int depcnt = (layer == 0) ? 1 : NT_F;

    uint l = get_sub_group_local_id(), sgl = get_sub_group_id();
    WaitForSemaphore_block(0, (volatile __global atomic_int*)(c->sync + dep), depcnt);
    __global half* h = c->h + c->tok_off;
    float rms = wg_rms(h, slm);
    for (int g = 0; g < TF; g++) {
        uint gi = (uint)tile*(SGN*TF) + (uint)g*SGN + sgl;
        uint n = gi*RPS;
        float o[RPS];
        sg_gemv_rms(h, c->wn+wn_off, rms, c->kw+kw_off, n, l, o);
        if (l==0) vstore2(convert_half2((float2)(o[0], o[1])), 0, c->kb+n);
    }

    // const __global half* vector = c->nbuf;
    // const __global half* matrix = c->kw + kw_off;
    // __global half* output = c->kb;
    // GemvBlock_1024x1024(tile, matrix, vector, output,
    //                 slm,
    //                 sem,
    //                 1);

    SignalSemaphore_block(0, sem);
}

inline void mk_stageAV(const MkTask t, __local char* slm) {
    __global const MonoCtx* c = t.ctx;
    int layer = t.layer, tile = t.tile;
    uint wn_off=layer*H, vw_off=layer*KVDIM*H;
    volatile __global atomic_int* sem = (volatile __global atomic_int*)(c->sync + layer*5 + 0);
    int dep = (layer == 0) ? EMBED_IDX : ((layer-1)*5 + 4);
    int depcnt = (layer == 0) ? 1 : NT_F;

    uint l = get_sub_group_local_id(), sgl = get_sub_group_id();
    WaitForSemaphore_block(0, (volatile __global atomic_int*)(c->sync + dep), depcnt);
    __global half* h = c->h + c->tok_off;
    float rms = wg_rms(h, slm);
    for (int g = 0; g < TF; g++) {
        uint gi = (uint)tile*(SGN*TF) + (uint)g*SGN + sgl;
        uint n = gi*RPS;
        float o[RPS];
        sg_gemv_rms(h, c->wn+wn_off, rms, c->vw+vw_off, n, l, o);
        if (l==0) vstore2(convert_half2((float2)(o[0], o[1])), 0, c->vb+n);
    }

    // const __global half* vector = c->nbuf;
    // const __global half* matrix = c->vw + vw_off;
    // __global half* output = c->vb;

    // GemvBlock_1024x1024(tile, matrix, vector, output,
    //                 slm,
    //                 sem,
    //                 1);

    SignalSemaphore_block(0, sem);
}

// Stage BC: fused RoPE + flash-decoding attention. tile in [0,NH) is a query
// head; tile in [NH,NH+KVH) writes the current token's K/V to the cache.
inline void mk_stageBC(const MkTask t, __local char* slm) {
    __global const MonoCtx* c = t.ctx;
    int layer = t.layer; uint wg = (uint)t.tile;
    uint l = get_sub_group_local_id(), sgl = get_sub_group_id(), nsgl = get_num_sub_groups();
    WaitForSemaphore_block(0, (volatile __global atomic_int*)(c->sync + layer*5 + 0), NT_A);

    const float scl = rsqrt((float)HD);
    uint CS = c->CS; int pos = (int)c->past_pos[0] + c->step;
    uint qn_off=layer*HD, kn_off=layer*HD;
    __local float* lsm_m = (__local float*)slm;
    __local float* lsm_l = lsm_m + SGN;
    __local float (*lsm_a)[NPL][SG] = (__local float(*)[NPL][SG])(lsm_l + SGN);

    if (wg < NH) {
        uint hq=wg, kv=hq/GQA;
        __global half* qhs = c->qb + hq*HD;
        float4 qraw = convert_float4(as_half4(intel_sub_group_block_read_us4((const __global ushort*)qhs)));
        float4 qnorm = convert_float4(as_half4(intel_sub_group_block_read_us4(
            (const __global ushort*)(c->qn+qn_off))));
        float sq=dot(qraw,qraw), qv[NPL];
        float iq=rsqrt(sub_group_reduce_add(sq)/HD+EPS);
        for (int j=0;j<NPL;j++) qv[j]=qraw[j]*iq*qnorm[j];
        float qr[NPL];
        for (int j=0;j<NPL/2;j++){
            uint d=l+SG*j;
            float a=(float)pos*convert_float(c->rf[d]), cc=native_cos(a), sn=native_sin(a);
            float x0=qv[j], x1=qv[j+NPL/2];
            qr[j]=x0*cc-x1*sn; qr[j+NPL/2]=x1*cc+x0*sn;
        }
        ulong base=((ulong)layer*KVH+kv)*(ulong)CS*HD;
        uint tile=((uint)pos + nsgl - 1)/nsgl;
        uint s0=sgl*tile, s1=min(s0+tile,(uint)pos);
        float acc[NPL]; for (int j=0;j<NPL;j++) acc[j]=0;
        float m=-INFINITY, ls=0;
        for (uint s=s0;s<s1;s++){
            const __global ushort* kp = (const __global ushort*)(c->kc + base + (ulong)s*HD);
            const __global ushort* vp = (const __global ushort*)(c->vc + base + (ulong)s*HD);
            float4 kval = convert_float4(as_half4(intel_sub_group_block_read_us4(kp)));
            float4 vval = convert_float4(as_half4(intel_sub_group_block_read_us4(vp)));
            float pa=dot((float4)(qr[0],qr[1],qr[2],qr[3]),kval);
            float sc=sub_group_reduce_add(pa)*scl;
            float mn=fmax(m,sc), cr=native_exp(m-mn), p=native_exp(sc-mn);
            ls=ls*cr+p;
            for (int j=0;j<NPL;j++) acc[j]=acc[j]*cr+p*vval[j];
            m=mn;
        }
        if (sgl==0){
            __global half* khs = c->kb + kv*HD;
            float4 kraw = convert_float4(as_half4(intel_sub_group_block_read_us4((const __global ushort*)khs)));
            float4 knorm = convert_float4(as_half4(intel_sub_group_block_read_us4(
                (const __global ushort*)(c->kn+kn_off))));
            float sk=dot(kraw,kraw), kvv[NPL];
            float ik=rsqrt(sub_group_reduce_add(sk)/HD+EPS);
            for (int j=0;j<NPL;j++) kvv[j]=kraw[j]*ik*knorm[j];
            float kr[NPL];
            for (int j=0;j<NPL/2;j++){
                uint d=l+SG*j;
                float a=(float)pos*convert_float(c->rf[d]), cc=native_cos(a), sn=native_sin(a);
                float x0=kvv[j], x1=kvv[j+NPL/2];
                kr[j]=x0*cc-x1*sn; kr[j+NPL/2]=x1*cc+x0*sn;
            }
            float pa=0;
            for (int j=0;j<NPL;j++) pa+=qr[j]*kr[j];
            float sc=sub_group_reduce_add(pa)*scl;
            float mn=fmax(m,sc), cr=native_exp(m-mn), p=native_exp(sc-mn);
            ls=ls*cr+p;
            for (int j=0;j<NPL;j++) acc[j]=acc[j]*cr+p*convert_float(c->vb[kv*HD+l+SG*j]);
            m=mn;
        }
        if (l==0){ lsm_m[sgl]=m; lsm_l[sgl]=ls; }
        for (int j=0;j<NPL;j++) lsm_a[sgl][j][l]=acc[j];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sgl==0){
            float M=lsm_m[0], L=lsm_l[0], ac[NPL];
            for (int j=0;j<NPL;j++) ac[j]=lsm_a[0][j][l];
            for (uint tt=1;tt<nsgl;tt++){
                float mn=fmax(M,lsm_m[tt]), cr=native_exp(M-mn), p=native_exp(lsm_m[tt]-mn);
                L=L*cr+lsm_l[tt]*p;
                for (int j=0;j<NPL;j++) ac[j]=ac[j]*cr+lsm_a[tt][j][l]*p;
                M=mn;
            }
            float il=1.0f/L;
            for (int j=0;j<NPL;j++) c->xn[hq*HD+l+SG*j]=convert_half(ac[j]*il);
        }
    } else if (wg < NH+KVH) {
        uint kvh=wg-NH;
        if (sgl==0) {
            __global half* kh = c->kb + kvh*HD;
            float4 kraw = convert_float4(as_half4(intel_sub_group_block_read_us4((const __global ushort*)kh)));
            float4 knorm = convert_float4(as_half4(intel_sub_group_block_read_us4(
                (const __global ushort*)(c->kn+kn_off))));
            float sq=dot(kraw,kraw), kv2[NPL];
            float iv=rsqrt(sub_group_reduce_add(sq)/HD+EPS);
            for (int j=0;j<NPL;j++) kv2[j]=kraw[j]*iv*knorm[j];
            float ko[NPL];
            for (int j=0;j<NPL;j++) ko[j]=kv2[j];
            for (int j=0;j<NPL/2;j++){
                uint d=l+SG*j;
                float a=(float)pos*convert_float(c->rf[d]), cc=native_cos(a), sn=native_sin(a);
                float x0=kv2[j], x1=kv2[j+NPL/2];
                ko[j]=x0*cc-x1*sn; ko[j+NPL/2]=x1*cc+x0*sn;
            }
            ulong cbase=((ulong)layer*KVH+kvh)*(ulong)CS*HD + (ulong)pos*HD;
            for (int j=0;j<NPL;j++){
                uint e=l+SG*j;
                c->kc[cbase+e]=convert_half(ko[j]);
                c->vc[cbase+e]=c->vb[kvh*HD+e];
            }
        }
    }
    SignalSemaphore_block(0, (volatile __global atomic_int*)(c->sync + layer*5 + 1));
}

// Stage D: O-projection with residual add (h += xn . Wo).
inline void mk_stageD(const MkTask t, __local char* slm) {
    __global const MonoCtx* c = t.ctx;
    int layer = t.layer, tile = t.tile;
    uint l = get_sub_group_local_id(), sgl = get_sub_group_id();
    WaitForSemaphore_block(0, (volatile __global atomic_int*)(c->sync + layer*5 + 1), NT_BC);
    uint ow_off=layer*H*QDIM;
    __global half* h = c->h + c->tok_off;
    for (int g = 0; g < TF; g++) {
        uint gi = (uint)tile*(SGN*TF) + (uint)g*SGN + sgl;
        uint n = gi*RPS; float o[RPS];
        sg_gemv_f16(c->xn, c->ow+ow_off, QDIM, n, l, o);
        if (l==0) {
            float2 v = convert_float2(vload2(0, h+n)) + (float2)(o[0], o[1]);
            vstore2(convert_half2(v), 0, h+n);
        }
    }
    SignalSemaphore_block(0, (volatile __global atomic_int*)(c->sync + layer*5 + 2));
}

// Stage E: fused post-attn RMSNorm + gate/up + SiLU.
inline void mk_stageE(const MkTask t, __local char* slm) {
    __global const MonoCtx* c = t.ctx;
    int layer = t.layer, tile = t.tile;
    uint l = get_sub_group_local_id(), sgl = get_sub_group_id();
    WaitForSemaphore_block(0, (volatile __global atomic_int*)(c->sync + layer*5 + 2), NT_D);
    uint pn_off=layer*H, gw_off=layer*IM*H, uw_off=layer*IM*H;
    __global half* h = c->h + c->tok_off;
    float rms = wg_rms(h, slm);
    for (int g = 0; g < TF; g++) {
        uint gi = (uint)tile*(SGN*TF) + (uint)g*SGN + sgl;
        uint n = gi*RPS;
        float a[RPS], b[RPS];
        sg_gemv_rms(h, c->pn+pn_off, rms, c->gw+gw_off, n, l, a);
        sg_gemv_rms(h, c->pn+pn_off, rms, c->uw+uw_off, n, l, b);
        if (l==0) {
            float2 av = (float2)(a[0], a[1]), bv = (float2)(b[0], b[1]);
            vstore2(convert_half2((av/(1.0f+native_exp(-av)))*bv), 0, c->gbuf+n);
        }
    }
    SignalSemaphore_block(0, (volatile __global atomic_int*)(c->sync + layer*5 + 3));
}

// Stage F: down-projection with residual add (h += g . Wdown).
inline void mk_stageF(const MkTask t, __local char* slm) {
    __global const MonoCtx* c = t.ctx;
    int layer = t.layer, tile = t.tile;
    uint l = get_sub_group_local_id(), sgl = get_sub_group_id();
    WaitForSemaphore_block(0, (volatile __global atomic_int*)(c->sync + layer*5 + 3), NT_E);
    uint dw_off=layer*H*IM;
    __global half* h = c->h + c->tok_off;
    for (int g = 0; g < TF; g++) {
        uint gi = (uint)tile*(SGN*TF) + (uint)g*SGN + sgl;
        uint n = gi*RPS; float o[RPS];
        sg_gemv_f16(c->gbuf, c->dw+dw_off, IM, n, l, o);
        if (l==0) {
            float2 v = convert_float2(vload2(0, h+n)) + (float2)(o[0], o[1]);
            vstore2(convert_half2(v), 0, h+n);
            if (layer == NUM_L-1) vstore2(v, 0, c->out+c->tok_off+n);
        }
    }
    SignalSemaphore_block(0, (volatile __global atomic_int*)(c->sync + layer*5 + 4));
}

#include "common/inkernelProfile.hcl"

// Task dispatch: `type` selects the stage.
inline void ExecuteMkTask(TaskDesc task, __local char* slm) {
    const MkTask t = *(const MkTask*)task.payload;
    switch (task.type) {
        case 0: mk_embed(t, slm); break;
        // case 1: mk_normA(t, slm); break;
        case 2: mk_stageAQ(t, slm); break;
        case 3: mk_stageAK(t, slm); break;
        case 4: IN_KERNEL_PROFILE_BLOCK(mk_stageAV(t, slm), "mk_stageAV"); break;
        case 5: mk_stageBC(t, slm); break;
        case 6: mk_stageD(t, slm); break;
        case 7: mk_stageE(t, slm); break;
        case 8: mk_stageF(t, slm); break;
        default: break;
    }
}

#define WorkerMainLoop_block_EXEC_FUN ExecuteMkTask
#include "taskSystem/device/workerMainLoop_template.hcl"

__attribute__((reqd_work_group_size(THREADS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_task(__constant const TaskManager* taskManager) {
    _Static_assert(SLM_BYTES <= 64*1024, "SLM_BYTES exceeds device SLM capacity");
    __local char slm[64*1024];
    WorkerMainLoop_block(taskManager, slm);
}

// Batched B60 prefill path. Decode never dispatches these kernels.
typedef struct PrefillCtx {
    __global const half* hs;
    __global half* h;
    __global float* out;
    __global const half* wn; __global const half* pn;
    __global const half* qw; __global const half* kw; __global const half* vw;
    __global const half* ow; __global const half* gw; __global const half* uw;
    __global const half* dw;
    __global const half* qn; __global const half* kn; __global const half* rf;
    __global half* norm; __global half* qb; __global half* kb; __global half* vb;
    __global half* xn; __global half* gate; __global half* up; __global half* proj;
    __global half* scores;
    __global half* kc; __global half* vc;
    __global const long* positions;
    uint tokens; uint CS;
} PrefillCtx;

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefill_copy(__global const PrefillCtx* c) {
    uint index = get_global_id(0), count = c->tokens * H;
    if (index < count) c->h[index] = c->hs[index];
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefill_rms(__global const PrefillCtx* c, int layer, int post_attn) {
    uint token = get_group_id(0), lid = get_local_id(0);
    __global const half* input = c->h + token * H;
    __global const half* weight = (post_attn ? c->pn : c->wn) + layer * H;
    __local float sums[256];
    float sum = 0.0f;
    for (uint index = lid; index < H; index += 256) {
        float value = convert_float(input[index]);
        sum = fma(value, value, sum);
    }
    sums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 128; stride; stride >>= 1) {
        if (lid < stride) sums[lid] += sums[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = rsqrt(sums[0] / H + EPS);
    for (uint index = lid; index < H; index += 256)
        c->norm[token * H + index] = convert_half(convert_float(input[index]) * scale * convert_float(weight[index]));
}

// Sixteen SIMD16 subgroups compute a 16-token x 256-output tile. A 128-wide K
// panel is cooperatively staged in SLM, following oneDNN-style GEMM blocking.
// op: 0=Q, 1=K, 2=V, 3=O, 4=gate, 5=up, 6=down, 7=QKV, 8=gate+up.
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_gemm_m16(__global const PrefillCtx* c, int layer, int op) {
    uint lane = get_sub_group_local_id();
    uint output_index = get_group_id(0) * 256 + get_sub_group_id() * 16 + lane;
    uint token_base = get_group_id(1) * 16;
    uint input_dim, output_dim, output_stride;
    __global const half* input;
    __global const half* weight;
    __global half* output;
    if (op == 0) {
        input_dim = H; output_dim = QDIM; input = c->norm; weight = c->qw + (ulong)layer * QDIM * H; output = c->qb;
    } else if (op == 1) {
        input_dim = H; output_dim = KVDIM; input = c->norm; weight = c->kw + (ulong)layer * KVDIM * H; output = c->kb;
    } else if (op == 2) {
        input_dim = H; output_dim = KVDIM; input = c->norm; weight = c->vw + (ulong)layer * KVDIM * H; output = c->vb;
    } else if (op == 3) {
        input_dim = QDIM; output_dim = H; input = c->xn; weight = c->ow + (ulong)layer * H * QDIM; output = c->proj;
    } else if (op == 4) {
        input_dim = H; output_dim = IM; input = c->norm; weight = c->gw + (ulong)layer * IM * H; output = c->gate;
    } else if (op == 5) {
        input_dim = H; output_dim = IM; input = c->norm; weight = c->uw + (ulong)layer * IM * H; output = c->up;
    } else if (op == 6) {
        input_dim = IM; output_dim = H; input = c->gate; weight = c->dw + (ulong)layer * H * IM; output = c->proj;
    } else if (op == 7) {
        input_dim = H;
        input = c->norm;
        if (output_index < QDIM) {
            output_dim = QDIM; weight = c->qw + (ulong)layer * QDIM * H; output = c->qb;
        } else if (output_index < QDIM + KVDIM) {
            output_index -= QDIM;
            output_dim = KVDIM; weight = c->kw + (ulong)layer * KVDIM * H; output = c->kb;
        } else {
            output_index -= QDIM + KVDIM;
            output_dim = KVDIM; weight = c->vw + (ulong)layer * KVDIM * H; output = c->vb;
        }
    } else {
        input_dim = H;
        input = c->norm;
        if (output_index < IM) {
            output_dim = IM; weight = c->gw + (ulong)layer * IM * H; output = c->gate;
        } else {
            output_index -= IM;
            output_dim = IM; weight = c->uw + (ulong)layer * IM * H; output = c->up;
        }
    }
    output_stride = output_dim;
    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
    __local half input_panel[16 * 128];
    for (uint k_base = 0; k_base < input_dim; k_base += 128) {
        for (uint index = get_local_id(0); index < 16 * 128; index += 256) {
            uint token = index >> 7, feature = index & 127;
            input_panel[index] = token_base + token < c->tokens
                                     ? input[(token_base + token) * input_dim + k_base + feature]
                                     : (half)0.0h;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < 128; k += 16) {
            short8 a0 = (short8)(as_short(input_panel[0 * 128 + k + lane]),
                                 as_short(input_panel[1 * 128 + k + lane]),
                                 as_short(input_panel[2 * 128 + k + lane]),
                                 as_short(input_panel[3 * 128 + k + lane]),
                                 as_short(input_panel[4 * 128 + k + lane]),
                                 as_short(input_panel[5 * 128 + k + lane]),
                                 as_short(input_panel[6 * 128 + k + lane]),
                                 as_short(input_panel[7 * 128 + k + lane]));
            short8 a1 = (short8)(as_short(input_panel[8 * 128 + k + lane]),
                                 as_short(input_panel[9 * 128 + k + lane]),
                                 as_short(input_panel[10 * 128 + k + lane]),
                                 as_short(input_panel[11 * 128 + k + lane]),
                                 as_short(input_panel[12 * 128 + k + lane]),
                                 as_short(input_panel[13 * 128 + k + lane]),
                                 as_short(input_panel[14 * 128 + k + lane]),
                                 as_short(input_panel[15 * 128 + k + lane]));
            int8 b = output_index < output_dim
                         ? as_int8(vload16(0, weight + (ulong)output_index * input_dim + k_base + k))
                         : (int8)(0);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b, acc1);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (output_index < output_dim) {
        if (token_base + 0 < c->tokens) output[(token_base + 0) * output_stride + output_index] = convert_half(acc0.s0);
        if (token_base + 1 < c->tokens) output[(token_base + 1) * output_stride + output_index] = convert_half(acc0.s1);
        if (token_base + 2 < c->tokens) output[(token_base + 2) * output_stride + output_index] = convert_half(acc0.s2);
        if (token_base + 3 < c->tokens) output[(token_base + 3) * output_stride + output_index] = convert_half(acc0.s3);
        if (token_base + 4 < c->tokens) output[(token_base + 4) * output_stride + output_index] = convert_half(acc0.s4);
        if (token_base + 5 < c->tokens) output[(token_base + 5) * output_stride + output_index] = convert_half(acc0.s5);
        if (token_base + 6 < c->tokens) output[(token_base + 6) * output_stride + output_index] = convert_half(acc0.s6);
        if (token_base + 7 < c->tokens) output[(token_base + 7) * output_stride + output_index] = convert_half(acc0.s7);
        if (token_base + 8 < c->tokens) output[(token_base + 8) * output_stride + output_index] = convert_half(acc1.s0);
        if (token_base + 9 < c->tokens) output[(token_base + 9) * output_stride + output_index] = convert_half(acc1.s1);
        if (token_base + 10 < c->tokens) output[(token_base + 10) * output_stride + output_index] = convert_half(acc1.s2);
        if (token_base + 11 < c->tokens) output[(token_base + 11) * output_stride + output_index] = convert_half(acc1.s3);
        if (token_base + 12 < c->tokens) output[(token_base + 12) * output_stride + output_index] = convert_half(acc1.s4);
        if (token_base + 13 < c->tokens) output[(token_base + 13) * output_stride + output_index] = convert_half(acc1.s5);
        if (token_base + 14 < c->tokens) output[(token_base + 14) * output_stride + output_index] = convert_half(acc1.s6);
        if (token_base + 15 < c->tokens) output[(token_base + 15) * output_stride + output_index] = convert_half(acc1.s7);
    }
}

// Three accumulators trade M16's finer grid for fewer token work-groups on
// medium prompts without taking on the M32 specialization's register footprint.
#if 0
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_gemm_m24(__global const PrefillCtx* c, int layer, int op) {
    uint lane = get_sub_group_local_id();
    uint output_index = get_group_id(0) * 256 + get_sub_group_id() * 16 + lane;
    uint token_base = get_group_id(1) * 24;
    uint input_dim, output_dim, output_stride;
    __global const half* input;
    __global const half* weight;
    __global half* output;
    if (op == 0) {
        input_dim = H; output_dim = QDIM; input = c->norm; weight = c->qw + (ulong)layer * QDIM * H; output = c->qb;
    } else if (op == 1) {
        input_dim = H; output_dim = KVDIM; input = c->norm; weight = c->kw + (ulong)layer * KVDIM * H; output = c->kb;
    } else if (op == 2) {
        input_dim = H; output_dim = KVDIM; input = c->norm; weight = c->vw + (ulong)layer * KVDIM * H; output = c->vb;
    } else if (op == 3) {
        input_dim = QDIM; output_dim = H; input = c->xn; weight = c->ow + (ulong)layer * H * QDIM; output = c->proj;
    } else if (op == 4) {
        input_dim = H; output_dim = IM; input = c->norm; weight = c->gw + (ulong)layer * IM * H; output = c->gate;
    } else if (op == 5) {
        input_dim = H; output_dim = IM; input = c->norm; weight = c->uw + (ulong)layer * IM * H; output = c->up;
    } else if (op == 6) {
        input_dim = IM; output_dim = H; input = c->gate; weight = c->dw + (ulong)layer * H * IM; output = c->proj;
    } else if (op == 7) {
        input_dim = H;
        input = c->norm;
        if (output_index < QDIM) {
            output_dim = QDIM; weight = c->qw + (ulong)layer * QDIM * H; output = c->qb;
        } else if (output_index < QDIM + KVDIM) {
            output_index -= QDIM;
            output_dim = KVDIM; weight = c->kw + (ulong)layer * KVDIM * H; output = c->kb;
        } else {
            output_index -= QDIM + KVDIM;
            output_dim = KVDIM; weight = c->vw + (ulong)layer * KVDIM * H; output = c->vb;
        }
    } else {
        input_dim = H;
        input = c->norm;
        if (output_index < IM) {
            output_dim = IM; weight = c->gw + (ulong)layer * IM * H; output = c->gate;
        } else {
            output_index -= IM;
            output_dim = IM; weight = c->uw + (ulong)layer * IM * H; output = c->up;
        }
    }
    output_stride = output_dim;
    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f), acc2 = (float8)(0.0f);
    __local half input_panel[24 * 128];
    for (uint k_base = 0; k_base < input_dim; k_base += 128) {
        for (uint index = get_local_id(0); index < 24 * 128; index += 256) {
            uint token = index >> 7, feature = index & 127;
            input_panel[index] = token_base + token < c->tokens
                                     ? input[(token_base + token) * input_dim + k_base + feature]
                                     : (half)0.0h;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < 128; k += 16) {
            short8 a0 = (short8)(as_short(input_panel[0 * 128 + k + lane]),
                                 as_short(input_panel[1 * 128 + k + lane]),
                                 as_short(input_panel[2 * 128 + k + lane]),
                                 as_short(input_panel[3 * 128 + k + lane]),
                                 as_short(input_panel[4 * 128 + k + lane]),
                                 as_short(input_panel[5 * 128 + k + lane]),
                                 as_short(input_panel[6 * 128 + k + lane]),
                                 as_short(input_panel[7 * 128 + k + lane]));
            short8 a1 = (short8)(as_short(input_panel[8 * 128 + k + lane]),
                                 as_short(input_panel[9 * 128 + k + lane]),
                                 as_short(input_panel[10 * 128 + k + lane]),
                                 as_short(input_panel[11 * 128 + k + lane]),
                                 as_short(input_panel[12 * 128 + k + lane]),
                                 as_short(input_panel[13 * 128 + k + lane]),
                                 as_short(input_panel[14 * 128 + k + lane]),
                                 as_short(input_panel[15 * 128 + k + lane]));
            short8 a2 = (short8)(as_short(input_panel[16 * 128 + k + lane]),
                                 as_short(input_panel[17 * 128 + k + lane]),
                                 as_short(input_panel[18 * 128 + k + lane]),
                                 as_short(input_panel[19 * 128 + k + lane]),
                                 as_short(input_panel[20 * 128 + k + lane]),
                                 as_short(input_panel[21 * 128 + k + lane]),
                                 as_short(input_panel[22 * 128 + k + lane]),
                                 as_short(input_panel[23 * 128 + k + lane]));
            int8 b = output_index < output_dim
                         ? as_int8(vload16(0, weight + (ulong)output_index * input_dim + k_base + k))
                         : (int8)(0);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b, acc2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (output_index < output_dim) {
        if (token_base + 0 < c->tokens) output[(token_base + 0) * output_stride + output_index] = convert_half(acc0.s0);
        if (token_base + 1 < c->tokens) output[(token_base + 1) * output_stride + output_index] = convert_half(acc0.s1);
        if (token_base + 2 < c->tokens) output[(token_base + 2) * output_stride + output_index] = convert_half(acc0.s2);
        if (token_base + 3 < c->tokens) output[(token_base + 3) * output_stride + output_index] = convert_half(acc0.s3);
        if (token_base + 4 < c->tokens) output[(token_base + 4) * output_stride + output_index] = convert_half(acc0.s4);
        if (token_base + 5 < c->tokens) output[(token_base + 5) * output_stride + output_index] = convert_half(acc0.s5);
        if (token_base + 6 < c->tokens) output[(token_base + 6) * output_stride + output_index] = convert_half(acc0.s6);
        if (token_base + 7 < c->tokens) output[(token_base + 7) * output_stride + output_index] = convert_half(acc0.s7);
        if (token_base + 8 < c->tokens) output[(token_base + 8) * output_stride + output_index] = convert_half(acc1.s0);
        if (token_base + 9 < c->tokens) output[(token_base + 9) * output_stride + output_index] = convert_half(acc1.s1);
        if (token_base + 10 < c->tokens) output[(token_base + 10) * output_stride + output_index] = convert_half(acc1.s2);
        if (token_base + 11 < c->tokens) output[(token_base + 11) * output_stride + output_index] = convert_half(acc1.s3);
        if (token_base + 12 < c->tokens) output[(token_base + 12) * output_stride + output_index] = convert_half(acc1.s4);
        if (token_base + 13 < c->tokens) output[(token_base + 13) * output_stride + output_index] = convert_half(acc1.s5);
        if (token_base + 14 < c->tokens) output[(token_base + 14) * output_stride + output_index] = convert_half(acc1.s6);
        if (token_base + 15 < c->tokens) output[(token_base + 15) * output_stride + output_index] = convert_half(acc1.s7);
        if (token_base + 16 < c->tokens) output[(token_base + 16) * output_stride + output_index] = convert_half(acc2.s0);
        if (token_base + 17 < c->tokens) output[(token_base + 17) * output_stride + output_index] = convert_half(acc2.s1);
        if (token_base + 18 < c->tokens) output[(token_base + 18) * output_stride + output_index] = convert_half(acc2.s2);
        if (token_base + 19 < c->tokens) output[(token_base + 19) * output_stride + output_index] = convert_half(acc2.s3);
        if (token_base + 20 < c->tokens) output[(token_base + 20) * output_stride + output_index] = convert_half(acc2.s4);
        if (token_base + 21 < c->tokens) output[(token_base + 21) * output_stride + output_index] = convert_half(acc2.s5);
        if (token_base + 22 < c->tokens) output[(token_base + 22) * output_stride + output_index] = convert_half(acc2.s6);
        if (token_base + 23 < c->tokens) output[(token_base + 23) * output_stride + output_index] = convert_half(acc2.s7);
    }
}
#endif

// The long-prompt specialization reuses each weight operand across 32 token
// rows. Keeping it as a separate entry point avoids imposing four accumulators
// on the M16 kernel's register allocation.
__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_gemm_m32(__global const PrefillCtx* c, int layer, int op) {
    uint lane = get_sub_group_local_id();
    uint output_index = get_group_id(0) * 512 + get_sub_group_id() * 16 + lane;
    uint token_base = get_group_id(1) * 32;
    uint input_dim, output_dim, output_stride;
    __global const half* input;
    __global const half* weight;
    __global half* output;
    if (op == 0) {
        input_dim = H; output_dim = QDIM; input = c->norm; weight = c->qw + (ulong)layer * QDIM * H; output = c->qb;
    } else if (op == 1) {
        input_dim = H; output_dim = KVDIM; input = c->norm; weight = c->kw + (ulong)layer * KVDIM * H; output = c->kb;
    } else if (op == 2) {
        input_dim = H; output_dim = KVDIM; input = c->norm; weight = c->vw + (ulong)layer * KVDIM * H; output = c->vb;
    } else if (op == 3) {
        input_dim = QDIM; output_dim = H; input = c->xn; weight = c->ow + (ulong)layer * H * QDIM; output = c->proj;
    } else if (op == 4) {
        input_dim = H; output_dim = IM; input = c->norm; weight = c->gw + (ulong)layer * IM * H; output = c->gate;
    } else if (op == 5) {
        input_dim = H; output_dim = IM; input = c->norm; weight = c->uw + (ulong)layer * IM * H; output = c->up;
    } else if (op == 6) {
        input_dim = IM; output_dim = H; input = c->gate; weight = c->dw + (ulong)layer * H * IM; output = c->proj;
    } else if (op == 7) {
        input_dim = H;
        input = c->norm;
        if (output_index < QDIM) {
            output_dim = QDIM; weight = c->qw + (ulong)layer * QDIM * H; output = c->qb;
        } else if (output_index < QDIM + KVDIM) {
            output_index -= QDIM;
            output_dim = KVDIM; weight = c->kw + (ulong)layer * KVDIM * H; output = c->kb;
        } else {
            output_index -= QDIM + KVDIM;
            output_dim = KVDIM; weight = c->vw + (ulong)layer * KVDIM * H; output = c->vb;
        }
    } else {
        input_dim = H;
        input = c->norm;
        if (output_index < IM) {
            output_dim = IM; weight = c->gw + (ulong)layer * IM * H; output = c->gate;
        } else {
            output_index -= IM;
            output_dim = IM; weight = c->uw + (ulong)layer * IM * H; output = c->up;
        }
    }
    output_stride = output_dim;
    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f), acc3 = (float8)(0.0f);
    __local half input_panel[32 * 128];
    for (uint k_base = 0; k_base < input_dim; k_base += 128) {
        for (uint index = get_local_id(0); index < 32 * 128; index += 512) {
            uint token = index >> 7, feature = index & 127;
            input_panel[index] = token_base + token < c->tokens
                                     ? input[(token_base + token) * input_dim + k_base + feature]
                                     : (half)0.0h;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < 128; k += 16) {
            short8 a0 = (short8)(as_short(input_panel[0 * 128 + k + lane]),
                                 as_short(input_panel[1 * 128 + k + lane]),
                                 as_short(input_panel[2 * 128 + k + lane]),
                                 as_short(input_panel[3 * 128 + k + lane]),
                                 as_short(input_panel[4 * 128 + k + lane]),
                                 as_short(input_panel[5 * 128 + k + lane]),
                                 as_short(input_panel[6 * 128 + k + lane]),
                                 as_short(input_panel[7 * 128 + k + lane]));
            short8 a1 = (short8)(as_short(input_panel[8 * 128 + k + lane]),
                                 as_short(input_panel[9 * 128 + k + lane]),
                                 as_short(input_panel[10 * 128 + k + lane]),
                                 as_short(input_panel[11 * 128 + k + lane]),
                                 as_short(input_panel[12 * 128 + k + lane]),
                                 as_short(input_panel[13 * 128 + k + lane]),
                                 as_short(input_panel[14 * 128 + k + lane]),
                                 as_short(input_panel[15 * 128 + k + lane]));
            short8 a2 = (short8)(as_short(input_panel[16 * 128 + k + lane]),
                                 as_short(input_panel[17 * 128 + k + lane]),
                                 as_short(input_panel[18 * 128 + k + lane]),
                                 as_short(input_panel[19 * 128 + k + lane]),
                                 as_short(input_panel[20 * 128 + k + lane]),
                                 as_short(input_panel[21 * 128 + k + lane]),
                                 as_short(input_panel[22 * 128 + k + lane]),
                                 as_short(input_panel[23 * 128 + k + lane]));
            short8 a3 = (short8)(as_short(input_panel[24 * 128 + k + lane]),
                                 as_short(input_panel[25 * 128 + k + lane]),
                                 as_short(input_panel[26 * 128 + k + lane]),
                                 as_short(input_panel[27 * 128 + k + lane]),
                                 as_short(input_panel[28 * 128 + k + lane]),
                                 as_short(input_panel[29 * 128 + k + lane]),
                                 as_short(input_panel[30 * 128 + k + lane]),
                                 as_short(input_panel[31 * 128 + k + lane]));
            int8 b = output_index < output_dim
                         ? as_int8(vload16(0, weight + (ulong)output_index * input_dim + k_base + k))
                         : (int8)(0);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b, acc3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (output_index < output_dim) {
        if (token_base + 0 < c->tokens) output[(token_base + 0) * output_stride + output_index] = convert_half(acc0.s0);
        if (token_base + 1 < c->tokens) output[(token_base + 1) * output_stride + output_index] = convert_half(acc0.s1);
        if (token_base + 2 < c->tokens) output[(token_base + 2) * output_stride + output_index] = convert_half(acc0.s2);
        if (token_base + 3 < c->tokens) output[(token_base + 3) * output_stride + output_index] = convert_half(acc0.s3);
        if (token_base + 4 < c->tokens) output[(token_base + 4) * output_stride + output_index] = convert_half(acc0.s4);
        if (token_base + 5 < c->tokens) output[(token_base + 5) * output_stride + output_index] = convert_half(acc0.s5);
        if (token_base + 6 < c->tokens) output[(token_base + 6) * output_stride + output_index] = convert_half(acc0.s6);
        if (token_base + 7 < c->tokens) output[(token_base + 7) * output_stride + output_index] = convert_half(acc0.s7);
        if (token_base + 8 < c->tokens) output[(token_base + 8) * output_stride + output_index] = convert_half(acc1.s0);
        if (token_base + 9 < c->tokens) output[(token_base + 9) * output_stride + output_index] = convert_half(acc1.s1);
        if (token_base + 10 < c->tokens) output[(token_base + 10) * output_stride + output_index] = convert_half(acc1.s2);
        if (token_base + 11 < c->tokens) output[(token_base + 11) * output_stride + output_index] = convert_half(acc1.s3);
        if (token_base + 12 < c->tokens) output[(token_base + 12) * output_stride + output_index] = convert_half(acc1.s4);
        if (token_base + 13 < c->tokens) output[(token_base + 13) * output_stride + output_index] = convert_half(acc1.s5);
        if (token_base + 14 < c->tokens) output[(token_base + 14) * output_stride + output_index] = convert_half(acc1.s6);
        if (token_base + 15 < c->tokens) output[(token_base + 15) * output_stride + output_index] = convert_half(acc1.s7);
        if (token_base + 16 < c->tokens) output[(token_base + 16) * output_stride + output_index] = convert_half(acc2.s0);
        if (token_base + 17 < c->tokens) output[(token_base + 17) * output_stride + output_index] = convert_half(acc2.s1);
        if (token_base + 18 < c->tokens) output[(token_base + 18) * output_stride + output_index] = convert_half(acc2.s2);
        if (token_base + 19 < c->tokens) output[(token_base + 19) * output_stride + output_index] = convert_half(acc2.s3);
        if (token_base + 20 < c->tokens) output[(token_base + 20) * output_stride + output_index] = convert_half(acc2.s4);
        if (token_base + 21 < c->tokens) output[(token_base + 21) * output_stride + output_index] = convert_half(acc2.s5);
        if (token_base + 22 < c->tokens) output[(token_base + 22) * output_stride + output_index] = convert_half(acc2.s6);
        if (token_base + 23 < c->tokens) output[(token_base + 23) * output_stride + output_index] = convert_half(acc2.s7);
        if (token_base + 24 < c->tokens) output[(token_base + 24) * output_stride + output_index] = convert_half(acc3.s0);
        if (token_base + 25 < c->tokens) output[(token_base + 25) * output_stride + output_index] = convert_half(acc3.s1);
        if (token_base + 26 < c->tokens) output[(token_base + 26) * output_stride + output_index] = convert_half(acc3.s2);
        if (token_base + 27 < c->tokens) output[(token_base + 27) * output_stride + output_index] = convert_half(acc3.s3);
        if (token_base + 28 < c->tokens) output[(token_base + 28) * output_stride + output_index] = convert_half(acc3.s4);
        if (token_base + 29 < c->tokens) output[(token_base + 29) * output_stride + output_index] = convert_half(acc3.s5);
        if (token_base + 30 < c->tokens) output[(token_base + 30) * output_stride + output_index] = convert_half(acc3.s6);
        if (token_base + 31 < c->tokens) output[(token_base + 31) * output_stride + output_index] = convert_half(acc3.s7);
    }
}

// All NH+KVH heads for a token share one work-group (one subgroup per head)
// instead of one 16-thread work-group per head; this cuts the number of
// launched work-groups for this stage by 24x (see Iteration 29).
__attribute__((reqd_work_group_size((NH + KVH) * 16, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_rope_cache(__global const PrefillCtx* c, int layer) {
    uint lane = get_sub_group_local_id(), head = get_sub_group_id(), token = get_group_id(1);
    int pos = (int)c->positions[0] + token;
    if (head < NH) {
        __global half* query = c->qb + token * QDIM + head * HD;
        float values[8], sum = 0.0f;
        for (uint j = 0; j < 8; ++j) { values[j] = convert_float(query[lane + j * 16]); sum = fma(values[j], values[j], sum); }
        float scale = rsqrt(sub_group_reduce_add(sum) / HD + EPS);
        for (uint j = 0; j < 4; ++j) {
            uint dim = lane + j * 16;
            float angle = (float)pos * convert_float(c->rf[dim]), cs = native_cos(angle), sn = native_sin(angle);
            float lo = values[j] * scale * convert_float(c->qn[layer * HD + dim]);
            float hi = values[j + 4] * scale * convert_float(c->qn[layer * HD + dim + HHD]);
            query[dim] = convert_half(lo * cs - hi * sn);
            query[dim + HHD] = convert_half(hi * cs + lo * sn);
        }
    } else {
        uint kv_head = head - NH;
        __global half* key = c->kb + token * KVDIM + kv_head * HD;
        __global half* value = c->vb + token * KVDIM + kv_head * HD;
        float values[8], sum = 0.0f;
        for (uint j = 0; j < 8; ++j) { values[j] = convert_float(key[lane + j * 16]); sum = fma(values[j], values[j], sum); }
        float scale = rsqrt(sub_group_reduce_add(sum) / HD + EPS);
        ulong cache_base = ((ulong)layer * KVH + kv_head) * (ulong)c->CS * HD + (ulong)pos * HD;
        for (uint j = 0; j < 4; ++j) {
            uint dim = lane + j * 16;
            float angle = (float)pos * convert_float(c->rf[dim]), cs = native_cos(angle), sn = native_sin(angle);
            float lo = values[j] * scale * convert_float(c->kn[layer * HD + dim]);
            float hi = values[j + 4] * scale * convert_float(c->kn[layer * HD + dim + HHD]);
            c->kc[cache_base + dim] = convert_half(lo * cs - hi * sn);
            c->kc[cache_base + dim + HHD] = convert_half(hi * cs + lo * sn);
            c->vc[cache_base + dim] = value[dim];
            c->vc[cache_base + dim + HHD] = value[dim + HHD];
        }
    }
}

__attribute__((reqd_work_group_size(16, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_attention(__global const PrefillCtx* c, int layer) {
    uint lane = get_sub_group_local_id(), query_head = get_group_id(0), token = get_group_id(1);
    uint kv_head = query_head / GQA;
    int pos = (int)c->positions[0] + token;
    __global const half* query = c->qb + token * QDIM + query_head * HD;
    ulong cache_base = ((ulong)layer * KVH + kv_head) * (ulong)c->CS * HD;
    float q[8], acc[8];
    for (uint j = 0; j < 8; ++j) { q[j] = convert_float(query[lane + j * 16]); acc[j] = 0.0f; }
    float maximum = -INFINITY, denominator = 0.0f;
    for (int sequence = 0; sequence <= pos; ++sequence) {
        float dot_product = 0.0f;
        for (uint j = 0; j < 8; ++j)
            dot_product = fma(q[j], convert_float(c->kc[cache_base + (ulong)sequence * HD + lane + j * 16]), dot_product);
        float score = sub_group_reduce_add(dot_product) * rsqrt((float)HD);
        float new_maximum = fmax(maximum, score);
        float correction = native_exp(maximum - new_maximum), probability = native_exp(score - new_maximum);
        denominator = denominator * correction + probability;
        for (uint j = 0; j < 8; ++j) {
            float value = convert_float(c->vc[cache_base + (ulong)sequence * HD + lane + j * 16]);
            acc[j] = acc[j] * correction + probability * value;
        }
        maximum = new_maximum;
    }
    for (uint j = 0; j < 8; ++j)
        c->xn[token * QDIM + query_head * HD + lane + j * 16] = convert_half(acc[j] / denominator);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefill_attention_softmax(__global const PrefillCtx* c) {
    uint row = get_group_id(0), lid = get_local_id(0);
    uint token = row % c->tokens;
    __global half* scores = c->scores + (ulong)row * c->tokens;
    __local float reduction[256];
    float maximum = -INFINITY;
    for (uint key = lid; key <= token; key += 256)
        maximum = fmax(maximum, convert_float(scores[key]) * rsqrt((float)HD));
    reduction[lid] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 128; stride; stride >>= 1) {
        if (lid < stride) reduction[lid] = fmax(reduction[lid], reduction[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = reduction[0];
    float denominator = 0.0f;
    for (uint key = lid; key <= token; key += 256)
        denominator += native_exp(convert_float(scores[key]) * rsqrt((float)HD) - maximum);
    reduction[lid] = denominator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 128; stride; stride >>= 1) {
        if (lid < stride) reduction[lid] += reduction[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    denominator = reduction[0];
    for (uint key = lid; key < c->tokens; key += 256) {
        float probability = key <= token
                                ? native_exp(convert_float(scores[key]) * rsqrt((float)HD) - maximum) / denominator
                                : 0.0f;
        scores[key] = convert_half(probability);
    }
}

// op: 0=attention residual, 1=SiLU(gate)*up, 2=MLP residual/final output.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefill_elementwise(__global const PrefillCtx* c, int layer, int op) {
    uint index = get_global_id(0), width = op == 1 ? IM : H, count = c->tokens * width;
    if (index >= count) return;
    if (op == 0) {
        c->h[index] = convert_half(convert_float(c->h[index]) + convert_float(c->proj[index]));
    } else if (op == 1) {
        float gate = convert_float(c->gate[index]);
        c->gate[index] = convert_half(gate / (1.0f + native_exp(-gate)) * convert_float(c->up[index]));
    } else {
        float value = convert_float(c->h[index]) + convert_float(c->proj[index]);
        c->h[index] = convert_half(value);
        if (layer == NUM_L - 1) c->out[index] = value;
    }
}
)CL";

// ---------------------------------------------------------------------------
// Dispatch constants (tuned on Intel Arc Pro B60) — must mirror the kernel #defines
// ---------------------------------------------------------------------------
static constexpr int NUM_L = 28, H_DIM = 1024, KVH = 8, HD = 128;
static constexpr int NH = 16, IM_DIM = 3072;
static constexpr int QDIM = NH * HD, KVDIM = KVH * HD;
static constexpr int RPS = 2;
static constexpr int MAX_SEQ = 4096;  // capacity of the internal KV cache (per layer/head)

// Task-system tiling — mirror of the kernel macros (SGN=16, TF=1).
static constexpr int SGN = 16, TF = 1;
static constexpr int NT_AQ = QDIM / (RPS * SGN * TF);   // 64
static constexpr int NT_AK = KVDIM / (RPS * SGN * TF);  // 32
static constexpr int NT_AV = KVDIM / (RPS * SGN * TF);  // 32
static constexpr int NT_A = NT_AQ + NT_AK + NT_AV;      // 128
static constexpr int NT_BC = NH + KVH;                  // 24
static constexpr int NT_D = H_DIM / (RPS * SGN * TF);   // 32
static constexpr int NT_E = IM_DIM / (RPS * SGN * TF);  // 96
static constexpr int NT_F = H_DIM / (RPS * SGN * TF);   // 32
static constexpr int SYNC_N = NUM_L * 5 + 1;            // per-(layer,stage) counters + embed

// Task-worker launch geometry. Like the old monokernel grid, the worker pool
// must be co-resident on the device: a consumer task spin-waits on its producer
// stage's counter, so if a pulled task's producers are not actually scheduled the
// wait would never complete. The safe worker count depends on this (register- and
// SLM-heavy) kernel's occupancy, not the device max, so it is tunable via
// OV_MEGAKERNEL_MONO_WG (default). More workers add parallelism to the GEMV
// stages up to the co-residency cap; attention has NH+KVH=24 independent tasks.
// Tuned on the B60 (24 Xe-cores): with the LWS=512 (SGN=16) work-group,
// worker count controls co-resident GEMV
// parallelism (more outstanding weight loads => higher HBM utilisation) while
// staying below the point where shared-cursor atomic_inc contention dominates.
static constexpr int MONO_WG = 32, MONO_LWS = 512;

struct PrefillCtxH {
    void* hs = nullptr; void* h = nullptr; void* out = nullptr;
    void* wn = nullptr; void* pn = nullptr;
    void* qw = nullptr; void* kw = nullptr; void* vw = nullptr;
    void* ow = nullptr; void* gw = nullptr; void* uw = nullptr; void* dw = nullptr;
    void* qn = nullptr; void* kn = nullptr; void* rf = nullptr;
    void* norm = nullptr; void* qb = nullptr; void* kb = nullptr; void* vb = nullptr;
    void* xn = nullptr; void* gate = nullptr; void* up = nullptr; void* proj = nullptr;
    void* scores = nullptr;
    void* kc = nullptr; void* vc = nullptr;
    void* positions = nullptr;
    unsigned tokens = 0; unsigned CS = 0;
};

enum DnnlGemmShape { DNNL_Q, DNNL_KV, DNNL_O, DNNL_MLP, DNNL_DOWN, DNNL_GEMM_SHAPES };

struct DnnlGemmCall {
    dnnl::memory src;
    dnnl::memory weights;
    dnnl::memory dst;
    std::unordered_map<int, dnnl::memory> arguments;
};

struct DnnlPrefillPlan {
    std::array<dnnl::matmul, DNNL_GEMM_SHAPES> primitives;
    std::array<dnnl::memory::desc, DNNL_GEMM_SHAPES> src_descs;
    std::array<dnnl::memory::desc, DNNL_GEMM_SHAPES> weights_descs;
    std::array<dnnl::memory::desc, DNNL_GEMM_SHAPES> dst_descs;
    std::vector<DnnlGemmCall> calls;
    dnnl::matmul attention_qk;
    dnnl::matmul attention_pv;
    std::vector<DnnlGemmCall> attention_qk_calls;
    std::vector<DnnlGemmCall> attention_pv_calls;
};

struct PrefillState {
    cl_kernel copy = nullptr, rms = nullptr, gemm_m16 = nullptr, gemm_m32 = nullptr, rope = nullptr;
    cl_kernel attention = nullptr, elementwise = nullptr;
    cl_kernel attention_softmax = nullptr;
    clSetKernelArgMemPointerINTEL_fn set_usm_arg = nullptr;
    void* context = nullptr; void* norm = nullptr; void* qb = nullptr; void* kb = nullptr;
    void* vb = nullptr; void* xn = nullptr; void* gate = nullptr; void* up = nullptr; void* proj = nullptr;
    void* scores = nullptr;
    size_t scores_bytes = 0;
    PrefillCtxH host_context{};
    std::unique_ptr<dnnl::engine> dnnl_engine;
    std::unique_ptr<dnnl::stream> dnnl_stream;
    std::unordered_map<unsigned, std::unique_ptr<DnnlPrefillPlan>> dnnl_plans;
};

static std::unordered_map<const Qwen06BPOCRuntime*, PrefillState> gPrefillStates;

struct MkTaskH {
    void* ctx;
    int layer;
    int tile;
};

TErrorcode Qwen06BPOCRuntime::Init(const IConstantParams* constantParams, const IPlatformParams* platformParams) {
    const auto* platformParams_ = static_cast<const Qwen06BPlatformParams*>(platformParams);
    ctx_ = platformParams_->context;
    dev_ = platformParams_->deviceId;
    stream_ = platformParams_->stream;

    cl_platform_id platform = nullptr;
    clGetDeviceInfo(dev_, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
    usmAlloc_ = reinterpret_cast<clDeviceMemAllocINTEL_fn>(clGetExtensionFunctionAddressForPlatform(platform, "clDeviceMemAllocINTEL"));
    usmFree_ = reinterpret_cast<clMemFreeINTEL_fn>(clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL"));
    usmMemcpy_ = reinterpret_cast<clEnqueueMemcpyINTEL_fn>(clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemcpyINTEL"));
    usmMemFill_ = reinterpret_cast<clEnqueueMemFillINTEL_fn>(clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemFillINTEL"));
    assert_or_throw(usmAlloc_ && usmFree_ && usmMemcpy_ && usmMemFill_,
                    "[MegaKernel] Intel USM extension functions are unavailable");

    cl_int err;
    prog_ = clCreateProgramWithSource(ctx_, 1, &kKernelSrc, nullptr, &err);
    assert_or_throw(err == CL_SUCCESS, "[MegaKernel] clCreateProgramWithSource: ", err);
    const std::string build_options = std::string("-cl-std=CL3.0 -I ") + TASK_SYSTEM_OPENCL_ROOT;  //+
                                                                                                   //" -igc_opts 'VISAOptions=-hybridRAWithSpill'";
    err = clBuildProgram(prog_, 1, &dev_, build_options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::vector<char> log(n);
        clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        throw_error("[MegaKernel] Build failed:\n", std::string(log.begin(), log.end()));
    }

    kTask_ = clCreateKernel(prog_, "mk_task", &err);
    assert_or_throw(err == CL_SUCCESS, "[MegaKernel] clCreateKernel(mk_task): ", err);

    auto ualloc = [&](size_t bytes) -> void* {
        cl_int st = CL_SUCCESS;
        void* p = usmAlloc_(ctx_, dev_, nullptr, bytes, 0, &st);
        assert_or_throw(st == CL_SUCCESS && p, "[MegaKernel] USM device alloc: ", st);
        return p;
    };
    // Per-token scratch (reused across tokens; tokens are serialised).
    mQb_ = ualloc(QDIM * 2);
    mKb_ = ualloc(KVDIM * 2);
    mVb_ = ualloc(KVDIM * 2);
    mGb_ = ualloc(IM_DIM * 2);
    mXn_ = ualloc(QDIM * 2);
    mNb_ = ualloc(H_DIM * 2);
    mH_ = ualloc((size_t)MAX_SEQ * H_DIM * 2);
    // Persistent internal KV cache: [NUM_L, KVH, MAX_SEQ, HD] half, K and V.
    mKC_ = ualloc((size_t)NUM_L * KVH * MAX_SEQ * HD * 2);
    mVC_ = ualloc((size_t)NUM_L * KVH * MAX_SEQ * HD * 2);
    // Per-(layer,stage) completion counters (+ embedding). Reset each launch.
    mSync_ = ualloc(SYNC_N * sizeof(int));
    // Shared per-token context.
    mCtx_ = ualloc(sizeof(MonoCtxH));

    // Optimization log (B60, 19 / 58 / 281 token prompts):
    // Baseline reused decode per token: 59 / 170 / 821 ms, or 0.20x / 0.10x /
    // 0.05x versus standard OpenVINO. Iteration 1 batched all prompt tokens and
    // mapped every dense projection to SIMD16 XMX: 21 / 34 / 110 ms. Iteration
    // 2 shared each activation tile through 48 KiB SLM; 20 / 42 / 149 ms showed
    // that reduced occupancy outweighed reuse, so it was reverted. Iteration 3
    // fused QKV and gate/up launch families: 18 / 32 / 111 ms, retained because
    // it improves short/medium prompts without changing long-prompt performance.
    // Iterations 4-13 established K128/N256 and a prompt-size crossover between
    // M16 and M32. Iteration 14 split those shapes into independent entry points,
    // avoiding M32 register pressure on short prompts. Iterations 15-18 rejected
    // M24, K256, and N384; N512 improved the long M32 path. The retained dispatch
    // is M16/N256/K128 through 64 tokens and M32/N512/K128 above 64 tokens.
    // Iterations 19-23 rejected subgroup RMS reduction, N768/N1024, K64, and
    // a 32-token M32 threshold; these confirmed GEMM weight traffic as the
    // remaining bottleneck rather than reduction, SLM capacity, or barriers.
    // Iterations 24-28 repacked down, gate/up, QKV, and output weights in Init,
    // then tested transposed 16x16 tiles with subgroup block reads. All variants
    // were slower or neutral and failed long generated-output parity, so the
    // original row-major prefill weights remain retained without duplication.
    PrefillState prefill;
    auto create_prefill_kernel = [&](const char* name) {
        cl_kernel kernel = clCreateKernel(prog_, name, &err);
        assert_or_throw(err == CL_SUCCESS, "[MegaKernel] clCreateKernel(", name, "): ", err);
        return kernel;
    };
    prefill.copy = create_prefill_kernel("prefill_copy");
    prefill.rms = create_prefill_kernel("prefill_rms");
    prefill.gemm_m16 = create_prefill_kernel("prefill_gemm_m16");
    prefill.gemm_m32 = create_prefill_kernel("prefill_gemm_m32");
    prefill.rope = create_prefill_kernel("prefill_rope_cache");
    prefill.attention = create_prefill_kernel("prefill_attention");
    prefill.attention_softmax = create_prefill_kernel("prefill_attention_softmax");
    prefill.elementwise = create_prefill_kernel("prefill_elementwise");
    prefill.set_usm_arg = reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(platform, "clSetKernelArgMemPointerINTEL"));
    assert_or_throw(prefill.set_usm_arg, "[MegaKernel] clSetKernelArgMemPointerINTEL unavailable");
    prefill.context = ualloc(sizeof(PrefillCtxH));
    prefill.norm = ualloc((size_t)MAX_SEQ * H_DIM * 2);
    prefill.qb = ualloc((size_t)MAX_SEQ * QDIM * 2);
    prefill.kb = ualloc((size_t)MAX_SEQ * KVDIM * 2);
    prefill.vb = ualloc((size_t)MAX_SEQ * KVDIM * 2);
    prefill.xn = ualloc((size_t)MAX_SEQ * QDIM * 2);
    prefill.gate = ualloc((size_t)MAX_SEQ * IM_DIM * 2);
    prefill.up = ualloc((size_t)MAX_SEQ * IM_DIM * 2);
    prefill.proj = ualloc((size_t)MAX_SEQ * H_DIM * 2);
    prefill.dnnl_engine = std::make_unique<dnnl::engine>(dnnl::ocl_interop::make_engine(dev_, ctx_));
    prefill.dnnl_stream =
        std::make_unique<dnnl::stream>(dnnl::ocl_interop::make_stream(*prefill.dnnl_engine, stream_));
    for (cl_kernel kernel : {prefill.copy, prefill.rms, prefill.gemm_m16, prefill.gemm_m32, prefill.rope,
                             prefill.attention, prefill.attention_softmax, prefill.elementwise}) {
        assert_or_throw(prefill.set_usm_arg(kernel, 0, prefill.context) == CL_SUCCESS,
                        "[MegaKernel] set prefill context argument failed");
        cl_bool enable = CL_TRUE;
        clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL, sizeof(enable), &enable);
        clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL, sizeof(enable), &enable);
        clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL, sizeof(enable), &enable);
    }

    // Build the topologically-sorted task queue once. Each task carries the
    // context pointer plus its (layer, tile); the task `type` selects the stage.
    std::vector<TaskDesc> queue;
    auto push = [&](int type, int layer, int tile) {
        TaskDesc d{};
        d.type = type;
        MkTaskH t{};
        t.ctx = mCtx_;
        t.layer = layer;
        t.tile = tile;
        static_assert(sizeof(MkTaskH) <= sizeof(d.payload), "MkTask exceeds payload");
        std::memcpy(d.payload, &t, sizeof(t));
        queue.push_back(d);
    };
    push(0, 0, 0);  // embedding
    for (int L = 0; L < NUM_L; L++) {
        // push(1, L, 0);                                 // input RMS scale
        for (int i = 0; i < NT_AQ; i++)
            push(2, L, i);  // Stage AQ (Q)
        for (int i = 0; i < NT_AK; i++)
            push(3, L, i);  // Stage AK (K)
        for (int i = 0; i < NT_AV; i++)
            push(4, L, i);  // Stage AV (V)
        for (int i = 0; i < NT_BC; i++)
            push(5, L, i);  // Stage BC (attention)
        for (int i = 0; i < NT_D; i++)
            push(6, L, i);  // Stage D  (o-proj)
        for (int i = 0; i < NT_E; i++)
            push(7, L, i);  // Stage E  (gate/up)
        for (int i = 0; i < NT_F; i++)
            push(8, L, i);  // Stage F  (down)
    }
    err = HostInitalizeTaskSystem(taskManager_, queue, static_cast<int*>(mSync_), SYNC_N, dev_, ctx_, stream_);
    assert_or_throw(err == CL_SUCCESS, "[MegaKernel] task-system initialization failed: ", err);

    // TaskManager descriptor consumed by the kernel as a __constant buffer.
    mTaskMgr_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(taskManager_), &taskManager_, &err);
    assert_or_throw(err == CL_SUCCESS, "[MegaKernel] clCreateBuffer(taskManager): ", err);

    assert_or_throw(clSetKernelArg(kTask_, 0, sizeof(cl_mem), &mTaskMgr_) == CL_SUCCESS,
                    "[MegaKernel] set taskManager arg failed");
    // The tasks reach every data buffer through USM pointers held in the
    // context, so allow the kernel to indirectly access any USM allocation
    // (device / host / shared) regardless of how OpenVINO allocated the weights.
    cl_bool enable = CL_TRUE;
    clSetKernelExecInfo(kTask_, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL, sizeof(enable), &enable);
    clSetKernelExecInfo(kTask_, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL, sizeof(enable), &enable);
    clSetKernelExecInfo(kTask_, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL, sizeof(enable), &enable);

    const auto* weights = static_cast<const Qwen06BConstantParams*>(constantParams);

    runtimeContext_.h = mH_;
    runtimeContext_.qb = mQb_;
    runtimeContext_.kb = mKb_;
    runtimeContext_.vb = mVb_;
    runtimeContext_.xn = mXn_;
    runtimeContext_.gbuf = mGb_;
    runtimeContext_.nbuf = mNb_;
    runtimeContext_.kc = mKC_;
    runtimeContext_.vc = mVC_;
    runtimeContext_.sync = mSync_;
    runtimeContext_.CS = (unsigned)MAX_SEQ;
    runtimeContext_.qw = weights->q_proj_w;
    runtimeContext_.kw = weights->k_proj_w;
    runtimeContext_.vw = weights->v_proj_w;
    runtimeContext_.ow = weights->o_proj_w;
    runtimeContext_.gw = weights->gate_proj_w;
    runtimeContext_.uw = weights->up_proj_w;
    runtimeContext_.dw = weights->down_proj_w;
    runtimeContext_.wn = weights->input_ln_w;
    runtimeContext_.pn = weights->post_attn_ln_w;
    runtimeContext_.qn = weights->q_norm_w;
    runtimeContext_.kn = weights->k_norm_w;
    runtimeContext_.rf = weights->rope_inv_freq;

    prefill.host_context.h = mH_;
    prefill.host_context.wn = weights->input_ln_w;
    prefill.host_context.pn = weights->post_attn_ln_w;
    prefill.host_context.qw = weights->q_proj_w;
    prefill.host_context.kw = weights->k_proj_w;
    prefill.host_context.vw = weights->v_proj_w;
    prefill.host_context.ow = weights->o_proj_w;
    prefill.host_context.gw = weights->gate_proj_w;
    prefill.host_context.uw = weights->up_proj_w;
    prefill.host_context.dw = weights->down_proj_w;
    prefill.host_context.qn = weights->q_norm_w;
    prefill.host_context.kn = weights->k_norm_w;
    prefill.host_context.rf = weights->rope_inv_freq;
    prefill.host_context.norm = prefill.norm;
    prefill.host_context.qb = prefill.qb;
    prefill.host_context.kb = prefill.kb;
    prefill.host_context.vb = prefill.vb;
    prefill.host_context.xn = prefill.xn;
    prefill.host_context.gate = prefill.gate;
    prefill.host_context.up = prefill.up;
    prefill.host_context.proj = prefill.proj;
    prefill.host_context.kc = mKC_;
    prefill.host_context.vc = mVC_;
    prefill.host_context.CS = MAX_SEQ;
    gPrefillStates.emplace(this, std::move(prefill));
    return 0;
}

TErrorcode Qwen06BPOCRuntime::Execute(const IRuntimeParams* runtimeParams) {
    const auto* io = static_cast<const Qwen06BRuntimeParams*>(runtimeParams);
    runtimeContext_.hs = io->hidden_states;
    runtimeContext_.past_pos = io->position_ids;
    runtimeContext_.out = io->hidden_states_out;
    const uint newTokens = io->newTokens;

    // Co-resident worker count (see MONO_WG note). Tunable via env.
    static const int workers = [] {
        const char* v = std::getenv("OV_MEGAKERNEL_MONO_WG");
        int w = v ? atoi(v) : MONO_WG;
        if (w < 1)
            w = 1;
        if (w > 160)
            w = 160;
        return w;
    }();

    if (newTokens > 1) {
        assert_or_throw(newTokens <= MAX_SEQ, "[MegaKernel] prefill exceeds KV-cache capacity: ", newTokens);
        auto state_it = gPrefillStates.find(this);
        assert_or_throw(state_it != gPrefillStates.end(), "[MegaKernel] prefill state is unavailable");
        PrefillState& prefill = state_it->second;
        const bool use_batched_attention = newTokens > 64 &&
                           std::getenv("OV_MEGAKERNEL_PREFILL_BATCHED_ATTENTION") == nullptr;
        if (use_batched_attention) {
            const size_t required_scores_bytes = (size_t)NH * newTokens * newTokens * sizeof(cl_half);
            if (required_scores_bytes > prefill.scores_bytes) {
                if (prefill.scores)
                    usmFree_(ctx_, prefill.scores);
                cl_int status = CL_SUCCESS;
                prefill.scores = usmAlloc_(ctx_, dev_, nullptr, required_scores_bytes, 0, &status);
                assert_or_throw(status == CL_SUCCESS && prefill.scores,
                                "[MegaKernel] batched attention score allocation failed: ", status);
                prefill.scores_bytes = required_scores_bytes;
                prefill.dnnl_plans.clear();
            }
        }
        prefill.host_context.hs = io->hidden_states;
        prefill.host_context.out = io->hidden_states_out;
        prefill.host_context.positions = io->position_ids;
        prefill.host_context.scores = prefill.scores;
        prefill.host_context.tokens = newTokens;
        assert_or_throw(usmMemcpy_(stream_, CL_TRUE, prefill.context, &prefill.host_context,
                                  sizeof(prefill.host_context), 0, nullptr, nullptr) == CL_SUCCESS,
                        "[MegaKernel] prefill context update failed");

        auto set_int_arg = [](cl_kernel kernel, cl_uint index, int value) {
            assert_or_throw(clSetKernelArg(kernel, index, sizeof(value), &value) == CL_SUCCESS,
                            "[MegaKernel] set prefill scalar argument failed");
        };
        int dispatch_index = 0;
        const bool sync_prefill = std::getenv("OV_MEGAKERNEL_PREFILL_SYNC") != nullptr;
        auto enqueue_1d = [&](cl_kernel kernel, size_t count, size_t local) {
            size_t global = (count + local - 1) / local * local;
            cl_int status = clEnqueueNDRangeKernel(stream_, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            assert_or_throw(status == CL_SUCCESS, "[MegaKernel] enqueue prefill kernel: ", status);
            ++dispatch_index;
            if (sync_prefill)
                assert_or_throw(clFinish(stream_) == CL_SUCCESS, "[MegaKernel] prefill dispatch failed: ", dispatch_index);
        };
        auto enqueue_2d = [&](cl_kernel kernel, size_t groups_x, size_t groups_y, size_t local_x) {
            size_t global[2] = {groups_x * local_x, groups_y};
            size_t local[2] = {local_x, 1};
            cl_int status = clEnqueueNDRangeKernel(stream_, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
            assert_or_throw(status == CL_SUCCESS, "[MegaKernel] enqueue prefill kernel: ", status);
            ++dispatch_index;
            if (sync_prefill)
                assert_or_throw(clFinish(stream_) == CL_SUCCESS, "[MegaKernel] prefill dispatch failed: ", dispatch_index);
        };
        auto gemm = [&](int layer, int op, int output_dim) {
            const bool use_m32 = newTokens > 64;
            cl_kernel kernel = use_m32 ? prefill.gemm_m32 : prefill.gemm_m16;
            const size_t token_tile = use_m32 ? 32 : 16;
            const size_t output_tile = use_m32 ? 512 : 256;
            set_int_arg(kernel, 1, layer);
            set_int_arg(kernel, 2, op);
            enqueue_2d(kernel,
                       (output_dim + output_tile - 1) / output_tile,
                       (newTokens + token_tile - 1) / token_tile,
                       output_tile);
        };

        const bool use_dnnl = std::getenv("OV_MEGAKERNEL_PREFILL_ONEDNN") == nullptr ||
                              std::strcmp(std::getenv("OV_MEGAKERNEL_PREFILL_ONEDNN"), "0") != 0;
        DnnlPrefillPlan* dnnl_plan = nullptr;
        if (use_dnnl) {
            auto plan_it = prefill.dnnl_plans.find(newTokens);
            if (plan_it == prefill.dnnl_plans.end()) {
                auto plan = std::make_unique<DnnlPrefillPlan>();
                dnnl::primitive_attr attr;
                auto create_shape = [&](DnnlGemmShape shape, int input_dim, int output_dim) {
                    plan->src_descs[shape] = dnnl::memory::desc(
                        {(dnnl::memory::dim)newTokens, input_dim}, dnnl::memory::data_type::f16,
                        {(dnnl::memory::dim)input_dim, 1});
                    plan->weights_descs[shape] = dnnl::memory::desc(
                        {input_dim, output_dim}, dnnl::memory::data_type::f16,
                        {1, (dnnl::memory::dim)input_dim});
                    plan->dst_descs[shape] = dnnl::memory::desc(
                        {(dnnl::memory::dim)newTokens, output_dim}, dnnl::memory::data_type::f16,
                        {(dnnl::memory::dim)output_dim, 1});
                    plan->primitives[shape] = dnnl::matmul(dnnl::matmul::primitive_desc(
                        *prefill.dnnl_engine, plan->src_descs[shape], plan->weights_descs[shape],
                        plan->dst_descs[shape], attr));
                };
                create_shape(DNNL_Q, H_DIM, QDIM);
                create_shape(DNNL_KV, H_DIM, KVDIM);
                create_shape(DNNL_O, QDIM, H_DIM);
                create_shape(DNNL_MLP, H_DIM, IM_DIM);
                create_shape(DNNL_DOWN, IM_DIM, H_DIM);

                plan->calls.reserve(NUM_L * 7);
                auto offset = [](void* base, size_t elements) {
                    return static_cast<void*>(static_cast<char*>(base) + elements * sizeof(cl_half));
                };
                auto add_call = [&](DnnlGemmShape shape, void* src, void* weights, void* dst) {
                    const auto kind = dnnl::ocl_interop::memory_kind::usm;
                    DnnlGemmCall call{dnnl::ocl_interop::make_memory(
                                          plan->src_descs[shape], *prefill.dnnl_engine, kind, {src}),
                                      dnnl::ocl_interop::make_memory(
                                          plan->weights_descs[shape], *prefill.dnnl_engine, kind, {weights}),
                                      dnnl::ocl_interop::make_memory(
                                          plan->dst_descs[shape], *prefill.dnnl_engine, kind, {dst}),
                                      {}};
                    call.arguments = {{DNNL_ARG_SRC, call.src},
                                      {DNNL_ARG_WEIGHTS, call.weights},
                                      {DNNL_ARG_DST, call.dst}};
                    plan->calls.push_back(std::move(call));
                };
                for (int layer = 0; layer < NUM_L; ++layer) {
                    add_call(DNNL_Q, prefill.norm, offset(prefill.host_context.qw, (size_t)layer * QDIM * H_DIM), prefill.qb);
                    add_call(DNNL_KV, prefill.norm, offset(prefill.host_context.kw, (size_t)layer * KVDIM * H_DIM), prefill.kb);
                    add_call(DNNL_KV, prefill.norm, offset(prefill.host_context.vw, (size_t)layer * KVDIM * H_DIM), prefill.vb);
                    add_call(DNNL_O, prefill.xn, offset(prefill.host_context.ow, (size_t)layer * H_DIM * QDIM), prefill.proj);
                    add_call(DNNL_MLP, prefill.norm, offset(prefill.host_context.gw, (size_t)layer * IM_DIM * H_DIM), prefill.gate);
                    add_call(DNNL_MLP, prefill.norm, offset(prefill.host_context.uw, (size_t)layer * IM_DIM * H_DIM), prefill.up);
                    add_call(DNNL_DOWN, prefill.gate, offset(prefill.host_context.dw, (size_t)layer * H_DIM * IM_DIM), prefill.proj);
                }
                if (use_batched_attention) {
                    using dim = dnnl::memory::dim;
                    const dim tokens = newTokens;
                    auto q_desc = dnnl::memory::desc({KVH, 2, tokens, HD}, dnnl::memory::data_type::f16,
                                                     {2 * HD, HD, QDIM, 1});
                    auto k_desc = dnnl::memory::desc({KVH, 1, HD, tokens}, dnnl::memory::data_type::f16,
                                                     {(dim)MAX_SEQ * HD, HD, 1, HD});
                    auto scores_desc = dnnl::memory::desc({KVH, 2, tokens, tokens}, dnnl::memory::data_type::f16,
                                                          {2 * tokens * tokens, tokens * tokens, tokens, 1});
                    auto v_desc = dnnl::memory::desc({KVH, 1, tokens, HD}, dnnl::memory::data_type::f16,
                                                     {(dim)MAX_SEQ * HD, HD, HD, 1});
                    auto output_desc = dnnl::memory::desc({KVH, 2, tokens, HD}, dnnl::memory::data_type::f16,
                                                          {2 * HD, HD, QDIM, 1});
                    plan->attention_qk = dnnl::matmul(dnnl::matmul::primitive_desc(
                        *prefill.dnnl_engine, q_desc, k_desc, scores_desc));
                    plan->attention_pv = dnnl::matmul(dnnl::matmul::primitive_desc(
                        *prefill.dnnl_engine, scores_desc, v_desc, output_desc));
                    const auto kind = dnnl::ocl_interop::memory_kind::usm;
                    auto add_attention_call = [&](std::vector<DnnlGemmCall>& calls,
                                                  const dnnl::memory::desc& src_desc, void* src,
                                                  const dnnl::memory::desc& weights_desc, void* weights,
                                                  const dnnl::memory::desc& dst_desc, void* dst) {
                        DnnlGemmCall call{dnnl::ocl_interop::make_memory(src_desc, *prefill.dnnl_engine, kind, {src}),
                                          dnnl::ocl_interop::make_memory(weights_desc, *prefill.dnnl_engine, kind, {weights}),
                                          dnnl::ocl_interop::make_memory(dst_desc, *prefill.dnnl_engine, kind, {dst}), {}};
                        call.arguments = {{DNNL_ARG_SRC, call.src}, {DNNL_ARG_WEIGHTS, call.weights}, {DNNL_ARG_DST, call.dst}};
                        calls.push_back(std::move(call));
                    };
                    for (int layer = 0; layer < NUM_L; ++layer) {
                        add_attention_call(plan->attention_qk_calls, q_desc, prefill.qb, k_desc,
                                           offset(prefill.host_context.kc, (size_t)layer * KVH * MAX_SEQ * HD),
                                           scores_desc, prefill.scores);
                        add_attention_call(plan->attention_pv_calls, scores_desc, prefill.scores, v_desc,
                                           offset(prefill.host_context.vc, (size_t)layer * KVH * MAX_SEQ * HD),
                                           output_desc, prefill.xn);
                    }
                }
                plan_it = prefill.dnnl_plans.emplace(newTokens, std::move(plan)).first;
            }
            dnnl_plan = plan_it->second.get();
        }
        auto dnnl_gemm = [&](int layer, int projection, DnnlGemmShape shape) {
            DnnlGemmCall& call = dnnl_plan->calls[layer * 7 + projection];
            dnnl_plan->primitives[shape].execute(*prefill.dnnl_stream, call.arguments);
        };

        enqueue_1d(prefill.copy, (size_t)newTokens * H_DIM, 256);
        for (int layer = 0; layer < NUM_L; ++layer) {
            set_int_arg(prefill.rms, 1, layer);
            set_int_arg(prefill.rms, 2, 0);
            enqueue_1d(prefill.rms, (size_t)newTokens * 256, 256);
            if (use_dnnl) {
                dnnl_gemm(layer, 0, DNNL_Q);
                dnnl_gemm(layer, 1, DNNL_KV);
                dnnl_gemm(layer, 2, DNNL_KV);
            } else {
                gemm(layer, 7, QDIM + 2 * KVDIM);
            }
            set_int_arg(prefill.rope, 1, layer);
            enqueue_2d(prefill.rope, 1, newTokens, (NH + KVH) * 16);
            set_int_arg(prefill.attention, 1, layer);
            if (use_batched_attention) {
                DnnlGemmCall& qk = dnnl_plan->attention_qk_calls[layer];
                dnnl_plan->attention_qk.execute(*prefill.dnnl_stream, qk.arguments);
                enqueue_1d(prefill.attention_softmax, (size_t)NH * newTokens * 256, 256);
                DnnlGemmCall& pv = dnnl_plan->attention_pv_calls[layer];
                dnnl_plan->attention_pv.execute(*prefill.dnnl_stream, pv.arguments);
            } else {
                enqueue_2d(prefill.attention, NH, newTokens, 16);
            }
            set_int_arg(prefill.elementwise, 1, layer);
            if (use_dnnl) dnnl_gemm(layer, 3, DNNL_O); else gemm(layer, 3, H_DIM);
            set_int_arg(prefill.elementwise, 2, 0);
            enqueue_1d(prefill.elementwise, (size_t)newTokens * H_DIM, 256);
            set_int_arg(prefill.rms, 2, 1);
            enqueue_1d(prefill.rms, (size_t)newTokens * 256, 256);
            if (use_dnnl) {
                dnnl_gemm(layer, 4, DNNL_MLP);
                dnnl_gemm(layer, 5, DNNL_MLP);
            } else {
                gemm(layer, 8, 2 * IM_DIM);
            }
            set_int_arg(prefill.elementwise, 2, 1);
            enqueue_1d(prefill.elementwise, (size_t)newTokens * IM_DIM, 256);
            if (use_dnnl) dnnl_gemm(layer, 6, DNNL_DOWN); else gemm(layer, 6, H_DIM);
            set_int_arg(prefill.elementwise, 2, 2);
            enqueue_1d(prefill.elementwise, (size_t)newTokens * H_DIM, 256);
        }
        return 0;
    }

    // ===== Task-system path: the whole 28-layer model as one queue of tasks,
    // launched once per new token. The in-order queue serialises tokens so
    // token t+1 sees the KV cache written by token t. For each token we refresh
    // the shared context (pos / token offset) and reset the sync counters, then
    // launch the worker pool which drains the queue.
    for (uint t = 0; t < newTokens; t++) {
        runtimeContext_.step = t;
        runtimeContext_.tok_off = t * (unsigned)H_DIM;
        assert_or_throw(usmMemcpy_(stream_, CL_TRUE, mCtx_, &runtimeContext_, sizeof(runtimeContext_), 0, nullptr, nullptr) == CL_SUCCESS,
                "[MegaKernel] context update failed");
        // Zero the stage counters and the FIFO cursor before the workers start.
        size_t g = (size_t)workers * MONO_LWS, l = (size_t)MONO_LWS;
        cl_int r = clEnqueueNDRangeKernel(stream_, kTask_, 1, nullptr, &g, &l, 0, nullptr, nullptr);
        assert_or_throw(r == CL_SUCCESS, "[MegaKernel] enqueue: ", r);
    }
    return 0;
}

TErrorcode Qwen06BPOCRuntime::Destroy() {
    auto state_it = gPrefillStates.find(this);
    if (state_it != gPrefillStates.end()) {
        PrefillState& prefill = state_it->second;
        prefill.dnnl_plans.clear();
        prefill.dnnl_stream.reset();
        prefill.dnnl_engine.reset();
        for (cl_kernel kernel : {prefill.copy, prefill.rms, prefill.gemm_m16, prefill.gemm_m32, prefill.rope,
                     prefill.attention, prefill.attention_softmax, prefill.elementwise}) {
            if (kernel)
                clReleaseKernel(kernel);
        }
        for (void* allocation : {prefill.context, prefill.norm, prefill.qb, prefill.kb,
                                 prefill.vb, prefill.xn, prefill.gate, prefill.up, prefill.proj}) {
            if (allocation && ctx_ && usmFree_)
                usmFree_(ctx_, allocation);
        }
        if (prefill.scores && ctx_ && usmFree_)
            usmFree_(ctx_, prefill.scores);
        gPrefillStates.erase(state_it);
    }

    if (ctx_ && dev_ && (taskManager_.workQueue || taskManager_.processedTaskCount)) {
        HostReleaseTaskSystem(taskManager_, dev_, ctx_);
        taskManager_ = {};
    }

    if (kTask_) {
        clReleaseKernel(kTask_);
        kTask_ = nullptr;
    }
    if (mTaskMgr_) {
        clReleaseMemObject(mTaskMgr_);
        mTaskMgr_ = nullptr;
    }
    if (prog_) {
        clReleaseProgram(prog_);
        prog_ = nullptr;
    }

    auto free_usm = [&](void*& allocation) {
        if (allocation && ctx_ && usmFree_) {
            usmFree_(ctx_, allocation);
            allocation = nullptr;
        }
    };

    free_usm(mQb_);
    free_usm(mKb_);
    free_usm(mVb_);
    free_usm(mGb_);
    free_usm(mXn_);
    free_usm(mH_);
    free_usm(mNb_);
    free_usm(mKC_);
    free_usm(mVC_);
    free_usm(mSync_);
    free_usm(mCtx_);

    ctx_ = nullptr;
    dev_ = nullptr;
    return 0;
}

}  // namespace mk