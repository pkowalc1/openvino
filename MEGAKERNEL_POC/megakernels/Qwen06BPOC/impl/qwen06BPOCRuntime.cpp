#include "qwen06BPOCRuntime.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
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
        return 0;
}

TErrorcode Qwen06BPOCRuntime::Execute(const IRuntimeParams* runtimeParams) {
    const auto* io = static_cast<const Qwen06BRuntimeParams*>(runtimeParams);
    runtimeContext_.hs = io->hidden_states;
    runtimeContext_.past_pos = io->position_ids;
    runtimeContext_.out = io->hidden_states_out;
    const uint newTokens = io->newTokens;

    // Take over a KV cache filled by someone else (the OpenVINO prefill path).
    // Source is [KVH, <per-tensor stride>, HD] per layer; destination [NUM_L, KVH, MAX_SEQ, HD].
    if (io->import_past && io->past_len > 0) {
        assert_or_throw(io->past_key && io->past_value && io->past_key_stride && io->past_value_stride,
                        "[MegaKernel] import_past without past pointers");
        assert_or_throw(io->past_len <= MAX_SEQ, "[MegaKernel] imported KV cache longer than MAX_SEQ");
        const size_t row_bytes = (size_t)io->past_len * HD * sizeof(unsigned short);
        for (int l = 0; l < NUM_L; l++) {
            const auto* src_k = static_cast<const unsigned short*>(io->past_key[l]);
            const auto* src_v = static_cast<const unsigned short*>(io->past_value[l]);
            assert_or_throw(src_k && src_v, "[MegaKernel] null KV pointer for layer ", l);
            for (int h = 0; h < KVH; h++) {
                const size_t dst_off = ((size_t)l * KVH + h) * MAX_SEQ * HD;
                assert_or_throw(usmMemcpy_(stream_, CL_FALSE, static_cast<unsigned short*>(mKC_) + dst_off,
                                           src_k + (size_t)h * io->past_key_stride[l] * HD,
                                           row_bytes, 0, nullptr, nullptr) == CL_SUCCESS,
                                "[MegaKernel] KV import (key) failed");
                assert_or_throw(usmMemcpy_(stream_, CL_FALSE, static_cast<unsigned short*>(mVC_) + dst_off,
                                           src_v + (size_t)h * io->past_value_stride[l] * HD,
                                           row_bytes, 0, nullptr, nullptr) == CL_SUCCESS,
                                "[MegaKernel] KV import (value) failed");
            }
        }
    }

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