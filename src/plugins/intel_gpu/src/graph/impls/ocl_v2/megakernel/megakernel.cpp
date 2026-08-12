// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MegaKernel plugin implementation — task-system-scheduled decoder.
// The entire Qwen3 decoder (all layers) runs in ONE kernel launch per token, but
// instead of a persistent grid + grid-wide software barrier, the work is driven
// by the fine-tuned GPU task system (MEGAKERNEL_POC/research/preloading_gemv):
// a pool of persistent worker work-groups pulls topologically-sorted tasks FIFO
// from a shared work queue; each layer stage is a set of per-workgroup tiles, and
// inter-stage ordering (the old grid barrier) is expressed as global atomic
// sync-flag dependencies resolved by the tasks themselves. Decode is one launch;
// prefill loops it per token. Key techniques: intel_sub_group_block_read, fused
// RMSNorm, fused RoPE, workgroup-cooperative flash-decoding attention, split-K GEMV.

#include "megakernel.hpp"
#include "intel_gpu/primitives/megakernel.hpp"
#include "megakernel_inst.h"
#include "../primitive_ocl_base.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/graph/network.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_memory.hpp"
#include "ocl/ocl_event.hpp"
#include "taskSystem/host/taskManagerHost.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace ov::intel_gpu::ocl {

using cldnn::ocl::ocl_engine;
using cldnn::ocl::ocl_stream;
using cldnn::ocl::ocl_event;
using cldnn::ocl::gpu_buffer;

namespace {

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
#define SG    16
#define RPS   4
#define NUM_L 28

// ---------------------------------------------------------------------------
// Barrier-free RMS: sub_group_reduce_add over each lane's strided H/SG slice
inline float sg_rms(const __global float* h, uint lane) {
    float ss = 0;
    for (uint k = lane * 16; k < H; k += SG * 16) {
        float16 v = vload16(0, h + k);
        float8  q = v.lo*v.lo + v.hi*v.hi;
        ss += q.s0+q.s1+q.s2+q.s3+q.s4+q.s5+q.s6+q.s7;
    }
    return rsqrt(sub_group_reduce_add(ss) / H + EPS);
}

// GEMV with fused RMS + block reads (SIMD16 message per 256-element strip)
inline void sg_gemv_rms(const __global float* h, const __global half* wn, float rms,
                        const __global half* w, uint base, uint lane, float* out) {
    float acc[RPS];
    for (int r = 0; r < RPS; r++) acc[r] = 0;
    for (uint blk = 0; blk < H; blk += SG * 16) {
        const __global uint*   hp = (const __global uint*)(h  + blk);
        float8 hlo = as_float8(intel_sub_group_block_read8(hp));
        float8 hhi = as_float8(intel_sub_group_block_read8(hp + SG * 8));
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
// TASK SYSTEM (verbatim logic from MEGAKERNEL_POC/research/preloading_gemv/src/
// ocl/taskSystem). Embedded here because the GPU plugin cannot include the
// research-tree headers; the scheduling logic, barriers and atomics are kept
// identical to the original taskManager.hcl / workerMainLoop_template.hcl.
// Workers (workgroups) pull topologically-sorted tasks FIFO from a shared work
// queue and execute them cooperatively; inter-task data dependencies are
// resolved by the tasks themselves through global atomic sync flags.
// ===========================================================================
#define PAYLOAD_SIZE (64 - sizeof(int))
typedef struct TaskDesc {
    int  type;
    char payload[PAYLOAD_SIZE];
} TaskDesc;

typedef struct TaskManager {
    __global const TaskDesc* workQueue;
    __global int*            processedTaskCount;
    int                      workQueueSize;
} TaskManager;

// GetNext task to execute (thread-level): atomically claim the next slot.
inline __global const TaskDesc* GetNextTask_thread(
    __constant const TaskManager* taskManager) {
    const int slotId = atomic_inc(taskManager->processedTaskCount);
    if (slotId >= taskManager->workQueueSize) {
        return NULL;
    }
    return taskManager->workQueue + slotId;
}

// GetNext task to execute (block-level): claim on thread 0 and broadcast the
// task pointer to the whole work-group through SLM.
inline __global const TaskDesc* GetNextTask_block(
    __constant const TaskManager* taskManager, __local char* slmBuffer) {
    __global const TaskDesc* task = NULL;
    __local ulong* taskAddress = (__local ulong*)slmBuffer;

    if (get_local_id(0) == 0) {
        task = GetNextTask_thread(taskManager);
        *taskAddress = (ulong)task;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    task = (__global const TaskDesc*)(*taskAddress);
    return task;
}

inline void ClearTaskManagerState_thread(
    __constant const TaskManager* taskManager) {
    atomic_xchg(taskManager->processedTaskCount, 0);
}

// Last executing worker restores the task manager state so the same queue can
// be re-launched (e.g. for the next token) without a host-side reset.
inline void LastWorkerClearTaskManagerState_block(
    __constant const TaskManager* taskManager) {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0) {
        volatile __global atomic_int* syncBuffer =
            (volatile __global atomic_int*)(taskManager->processedTaskCount);
        const int processed = atomic_load_explicit(syncBuffer, memory_order_acquire,
                                                    memory_scope_device);
        const int workers =
            get_num_groups(0) * get_num_groups(1) * get_num_groups(2);
        if (processed == workers + taskManager->workQueueSize) {
            ClearTaskManagerState_thread(taskManager);
        }
    }
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
#define NT_A ((QDIM+2*KVDIM)/(RPS*SGN*TF))     // Stage A (QKV) tile count      = 64
#define NT_BC (NH+KVH)                          // Stage BC (attn) task count    = 24
#define NT_D  (H/(RPS*SGN*TF))                  // Stage D (o-proj) tile count   = 16
#define NT_E  (IM/(RPS*SGN*TF))                 // Stage E (gate/up) tile count  = 48
#define NT_F  (H/(RPS*SGN*TF))                  // Stage F (down) tile count     = 16
#define EMBED_IDX (NUM_L*5)                     // sync slot for the embedding stage
#define SLM_BYTES ((2*SGN + SGN*NPL*SG)*4)      // flash-decoding partials (lsm_m/lsm_l/lsm_a)

// Per-token context shared by every task: all base pointers plus the scalars
// (pos, cache stride, token offset) that vary per launch. Lives in USM device
// memory; each task carries a pointer to it in its payload.
typedef struct MonoCtx {
    __global const half*  hs;
    __global float*       h;
    __global const half*  wn; __global const half* pn;
    __global const half*  qw; __global const half* kw; __global const half* vw;
    __global const half*  ow; __global const half* gw; __global const half* uw;
    __global const half*  dw;
    __global const half*  qn; __global const half* kn; __global const half* rf;
    __global float*       qb; __global float* kb; __global float* vb;
    __global half*        xn; __global half* gbuf;
    __global half*        kc; __global half* vc;
    __global int*         sync;
    int  pos; uint CS; uint tok_off;
} MonoCtx;

typedef struct MkTask {
    __global const MonoCtx* ctx;
    int layer;
    int tile;
} MkTask;

// Barrier-equivalent dependency wait: the whole work-group blocks until the
// producer stage's counter reaches `cnt` (all its tiles signalled). Mirrors the
// producer store-release / consumer load-acquire pattern of the task system.
inline void mk_wait(__global int* sync, int idx, int cnt) {
    // Entry barrier is LOCAL-only: at stage entry there is no producer global data
    // to make visible yet (that is the exit barrier's job, paired with the acquire
    // below), so a device-scope global fence + L1 flush here is pure overhead. A
    // local fence still orders the SLM task-pointer handoff from GetNextTask_block.
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        volatile __global atomic_int* p = (volatile __global atomic_int*)(sync + idx);
        // Poll with a RELAXED load so idle spinners do not force a device-scope
        // coherency transaction on every iteration (that traffic steals memory
        // bandwidth from the workers actually doing the stage). Once the counter
        // is observed to reach `cnt`, a single ACQUIRE load synchronizes-with the
        // producers' release stores so their data writes are visible.
        while (atomic_load_explicit(p, memory_order_relaxed, memory_scope_device) < cnt) {}
        atomic_load_explicit(p, memory_order_acquire, memory_scope_device);
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

// Signal completion of one tile of a stage (publishes this WG's global stores).
inline void mk_signal(__global int* sync, int idx) {
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        atomic_fetch_add_explicit((volatile __global atomic_int*)(sync + idx), 1,
                                  memory_order_release, memory_scope_device);
    }
}

// Stage 0: input embedding fp16 -> fp32 residual stream (single task).
inline void mk_embed(const MkTask* t, __local char* slm) {
    __global const MonoCtx* c = t->ctx;
    __global const half* hs = c->hs + c->tok_off;
    __global float*      h  = c->h  + c->tok_off;
    for (uint i = get_local_id(0); i < H; i += get_local_size(0))
        h[i] = convert_float(hs[i]);
    mk_signal(c->sync, EMBED_IDX);
}

// Stage A: fused input RMSNorm + Q/K/V projection (one tile of QKV rows).
inline void mk_stageA(const MkTask* t, __local char* slm) {
    __global const MonoCtx* c = t->ctx;
    int layer = t->layer, tile = t->tile;
    uint l = get_sub_group_local_id(), sgl = get_sub_group_id();
    int dep    = (layer == 0) ? EMBED_IDX : ((layer-1)*5 + 4);
    int depcnt = (layer == 0) ? 1         : NT_F;
    mk_wait(c->sync, dep, depcnt);

    uint wn_off=layer*H, qw_off=layer*QDIM*H, kw_off=layer*KVDIM*H, vw_off=layer*KVDIM*H;
    __global float* h = c->h + c->tok_off;
    for (int g = 0; g < TF; g++) {
        uint gi = (uint)tile*(SGN*TF) + (uint)g*SGN + sgl;
        uint gr = gi*RPS;
        float rms = sg_rms(h, l), o[RPS];
        if (gr < QDIM) {
            sg_gemv_rms(h, c->wn+wn_off, rms, c->qw+qw_off, gr, l, o);
            if (l==0) for (int r=0;r<RPS;r++) c->qb[gr+r]=o[r];
        } else if (gr < QDIM+KVDIM) {
            uint n=gr-QDIM;
            sg_gemv_rms(h, c->wn+wn_off, rms, c->kw+kw_off, n, l, o);
            if (l==0) for (int r=0;r<RPS;r++) c->kb[n+r]=o[r];
        } else {
            uint n=gr-QDIM-KVDIM;
            sg_gemv_rms(h, c->wn+wn_off, rms, c->vw+vw_off, n, l, o);
            if (l==0) for (int r=0;r<RPS;r++) c->vb[n+r]=o[r];
        }
    }
    mk_signal(c->sync, layer*5 + 0);
}

// Stage BC: fused RoPE + flash-decoding attention. tile in [0,NH) is a query
// head; tile in [NH,NH+KVH) writes the current token's K/V to the cache.
inline void mk_stageBC(const MkTask* t, __local char* slm) {
    __global const MonoCtx* c = t->ctx;
    int layer = t->layer; uint wg = (uint)t->tile;
    uint l = get_sub_group_local_id(), sgl = get_sub_group_id(), nsgl = get_num_sub_groups();
    mk_wait(c->sync, layer*5 + 0, NT_A);

    const float scl = rsqrt((float)HD);
    uint CS = c->CS; int pos = c->pos;
    uint qn_off=layer*HD, kn_off=layer*HD;
    __local float* lsm_m = (__local float*)slm;
    __local float* lsm_l = lsm_m + SGN;
    __local float (*lsm_a)[NPL][SG] = (__local float(*)[NPL][SG])(lsm_l + SGN);

    if (wg < NH) {
        uint hq=wg, kv=hq/GQA;
        __global float* qhs = c->qb + hq*HD;
        float sq=0, qv[8];
        for (int j=0;j<8;j++){ float x=qhs[l+SG*j]; qv[j]=x; sq+=x*x; }
        float iq=rsqrt(sub_group_reduce_add(sq)/HD+EPS);
        for (int j=0;j<8;j++) qv[j]=qv[j]*iq*convert_float(c->qn[qn_off+l+SG*j]);
        float qr[NPL];
        for (int j=0;j<4;j++){
            uint d=l+SG*j;
            float a=(float)pos*convert_float(c->rf[d]), cc=native_cos(a), sn=native_sin(a);
            float x0=qv[j], x1=qv[j+4];
            qr[j]=x0*cc-x1*sn; qr[j+4]=x1*cc+x0*sn;
        }
        ulong base=((ulong)layer*KVH+kv)*(ulong)CS*HD;
        uint tile=((uint)pos + nsgl - 1)/nsgl;
        uint s0=sgl*tile, s1=min(s0+tile,(uint)pos);
        float acc[NPL]; for (int j=0;j<NPL;j++) acc[j]=0;
        float m=-INFINITY, ls=0;
        for (uint s=s0;s<s1;s++){
            float pa=0;
            for (int j=0;j<NPL;j++) pa+=qr[j]*convert_float(c->kc[base+(ulong)s*HD+l+SG*j]);
            float sc=sub_group_reduce_add(pa)*scl;
            float mn=fmax(m,sc), cr=native_exp(m-mn), p=native_exp(sc-mn);
            ls=ls*cr+p;
            for (int j=0;j<NPL;j++) acc[j]=acc[j]*cr+p*convert_float(c->vc[base+(ulong)s*HD+l+SG*j]);
            m=mn;
        }
        if (sgl==0){
            __global float* khs = c->kb + kv*HD;
            float sk=0, kvv[8];
            for (int j=0;j<8;j++){ float x=khs[l+SG*j]; kvv[j]=x; sk+=x*x; }
            float ik=rsqrt(sub_group_reduce_add(sk)/HD+EPS);
            for (int j=0;j<8;j++) kvv[j]=kvv[j]*ik*convert_float(c->kn[kn_off+l+SG*j]);
            float kr[NPL];
            for (int j=0;j<4;j++){
                uint d=l+SG*j;
                float a=(float)pos*convert_float(c->rf[d]), cc=native_cos(a), sn=native_sin(a);
                float x0=kvv[j], x1=kvv[j+4];
                kr[j]=x0*cc-x1*sn; kr[j+4]=x1*cc+x0*sn;
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
            __global float* kh = c->kb + kvh*HD;
            float sq=0, kv2[8];
            for (int j=0;j<8;j++){ float x=kh[l+SG*j]; kv2[j]=x; sq+=x*x; }
            float iv=rsqrt(sub_group_reduce_add(sq)/HD+EPS);
            for (int j=0;j<8;j++) kv2[j]=kv2[j]*iv*convert_float(c->kn[kn_off+l+SG*j]);
            float ko[8];
            for (int j=0;j<8;j++) ko[j]=kv2[j];
            for (int j=0;j<4;j++){
                uint d=l+SG*j;
                float a=(float)pos*convert_float(c->rf[d]), cc=native_cos(a), sn=native_sin(a);
                float x0=kv2[j], x1=kv2[j+4];
                ko[j]=x0*cc-x1*sn; ko[j+4]=x1*cc+x0*sn;
            }
            ulong cbase=((ulong)layer*KVH+kvh)*(ulong)CS*HD + (ulong)pos*HD;
            for (int j=0;j<8;j++){
                uint e=l+SG*j;
                c->kc[cbase+e]=convert_half(ko[j]);
                c->vc[cbase+e]=convert_half(c->vb[kvh*HD+e]);
            }
        }
    }
    mk_signal(c->sync, layer*5 + 1);
}

// Stage D: O-projection with residual add (h += xn . Wo).
inline void mk_stageD(const MkTask* t, __local char* slm) {
    __global const MonoCtx* c = t->ctx;
    int layer = t->layer, tile = t->tile;
    uint l = get_sub_group_local_id(), sgl = get_sub_group_id();
    mk_wait(c->sync, layer*5 + 1, NT_BC);
    uint ow_off=layer*H*QDIM;
    __global float* h = c->h + c->tok_off;
    for (int g = 0; g < TF; g++) {
        uint gi = (uint)tile*(SGN*TF) + (uint)g*SGN + sgl;
        uint n = gi*RPS; float o[RPS];
        sg_gemv_f16(c->xn, c->ow+ow_off, QDIM, n, l, o);
        if (l==0) for (int r=0;r<RPS;r++) h[n+r]+=o[r];
    }
    mk_signal(c->sync, layer*5 + 2);
}

// Stage E: fused post-attn RMSNorm + gate/up + SiLU.
inline void mk_stageE(const MkTask* t, __local char* slm) {
    __global const MonoCtx* c = t->ctx;
    int layer = t->layer, tile = t->tile;
    uint l = get_sub_group_local_id(), sgl = get_sub_group_id();
    mk_wait(c->sync, layer*5 + 2, NT_D);
    uint pn_off=layer*H, gw_off=layer*IM*H, uw_off=layer*IM*H;
    __global float* h = c->h + c->tok_off;
    for (int g = 0; g < TF; g++) {
        uint gi = (uint)tile*(SGN*TF) + (uint)g*SGN + sgl;
        uint n = gi*RPS;
        float rms=sg_rms(h,l), a[RPS], b[RPS];
        sg_gemv_rms(h, c->pn+pn_off, rms, c->gw+gw_off, n, l, a);
        sg_gemv_rms(h, c->pn+pn_off, rms, c->uw+uw_off, n, l, b);
        if (l==0) for (int r=0;r<RPS;r++)
            c->gbuf[n+r]=convert_half((a[r]/(1.0f+native_exp(-a[r])))*b[r]);
    }
    mk_signal(c->sync, layer*5 + 3);
}

// Stage F: down-projection with residual add (h += g . Wdown).
inline void mk_stageF(const MkTask* t, __local char* slm) {
    __global const MonoCtx* c = t->ctx;
    int layer = t->layer, tile = t->tile;
    uint l = get_sub_group_local_id(), sgl = get_sub_group_id();
    mk_wait(c->sync, layer*5 + 3, NT_E);
    uint dw_off=layer*H*IM;
    __global float* h = c->h + c->tok_off;
    for (int g = 0; g < TF; g++) {
        uint gi = (uint)tile*(SGN*TF) + (uint)g*SGN + sgl;
        uint n = gi*RPS; float o[RPS];
        sg_gemv_f16(c->gbuf, c->dw+dw_off, IM, n, l, o);
        if (l==0) for (int r=0;r<RPS;r++) h[n+r]+=o[r];
    }
    mk_signal(c->sync, layer*5 + 4);
}

// Task dispatch: `type` selects the stage.
inline void ExecuteMkTask(TaskDesc task, __local char* slm) {
    const MkTask* t = (const MkTask*)task.payload;
    switch (task.type) {
        case 0: mk_embed (t, slm); break;
        case 1: mk_stageA(t, slm); break;
        case 2: mk_stageBC(t, slm); break;
        case 3: mk_stageD(t, slm); break;
        case 4: mk_stageE(t, slm); break;
        case 5: mk_stageF(t, slm); break;
        default: break;
    }
}

// Worker main loop (verbatim logic from workerMainLoop_template.hcl): each
// work-group pulls tasks FIFO until the queue is drained, then the last worker
// clears the task-manager state so the queue can be re-launched next token.
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_task(__constant const TaskManager* taskManager) {
    __local char slm[SLM_BYTES];
    __global const TaskDesc* taskPtr = GetNextTask_block(taskManager, slm);
    while (taskPtr != NULL) {
        TaskDesc task = *taskPtr;
        ExecuteMkTask(task, slm);
        taskPtr = GetNextTask_block(taskManager, slm);
    }
    LastWorkerClearTaskManagerState_block(taskManager);
}
)CL";

// ---------------------------------------------------------------------------
// Dispatch constants (tuned on Intel Arc Pro B60) — must mirror the kernel #defines
// ---------------------------------------------------------------------------
static constexpr int  NUM_L = 28, H_DIM = 1024, KVH = 8, HD = 128;
static constexpr int  NH = 16, IM_DIM = 3072;
static constexpr int  QDIM = NH * HD, KVDIM = KVH * HD;
static constexpr int  RPS = 4;
static constexpr int  MAX_SEQ = 4096;  // capacity of the internal KV cache (per layer/head)

// Task-system tiling — mirror of the kernel macros (SGN=16, TF=1).
static constexpr int  SGN = 16, TF = 1;
static constexpr int  NT_A  = (QDIM + 2 * KVDIM) / (RPS * SGN * TF);   // 64
static constexpr int  NT_BC = NH + KVH;                                // 24
static constexpr int  NT_D  = H_DIM / (RPS * SGN * TF);                // 16
static constexpr int  NT_E  = IM_DIM / (RPS * SGN * TF);               // 48
static constexpr int  NT_F  = H_DIM / (RPS * SGN * TF);                // 16
static constexpr int  SYNC_N = NUM_L * 5 + 1;                          // per-(layer,stage) counters + embed

// Task-worker launch geometry. Like the old monokernel grid, the worker pool
// must be co-resident on the device: a consumer task spin-waits on its producer
// stage's counter, so if a pulled task's producers are not actually scheduled the
// wait would never complete. The safe worker count depends on this (register- and
// SLM-heavy) kernel's occupancy, not the device max, so it is tunable via
// OV_MEGAKERNEL_MONO_WG (default). More workers add parallelism to the GEMV
// stages up to the co-residency cap; attention has NH+KVH=24 independent tasks.
// Tuned on the B60 (24 Xe-cores): with the lighter LWS=256 (SGN=16) work-group,
// two work-groups fit per Xe-core, so 32 workers maximise co-resident GEMV
// parallelism (more outstanding weight loads => higher HBM utilisation) while
// staying below the point where shared-cursor atomic_inc contention dominates.
static constexpr int  MONO_WG = 32, MONO_LWS = 256;

struct MonoCtxH {
    void* hs; void* h;
    void* wn; void* pn;
    void* qw; void* kw; void* vw;
    void* ow; void* gw; void* uw;
    void* dw;
    void* qn; void* kn; void* rf;
    void* qb; void* kb; void* vb;
    void* xn; void* gbuf;
    void* kc; void* vc;
    void* sync;
    int   pos; unsigned CS; unsigned tok_off;
};
struct MkTaskH {
    void* ctx;
    int   layer;
    int   tile;
};

// ---------------------------------------------------------------------------
// MegaKernelFastImpl
// ---------------------------------------------------------------------------

class MegaKernelFastImpl : public cldnn::primitive_impl {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MegaKernelFastImpl)

    MegaKernelFastImpl() = default;
    explicit MegaKernelFastImpl(const cldnn::program_node&, const RuntimeParams&) {}
    // Copy constructor: copy the primitive_impl base subobject so metadata such
    // as m_manager and the dynamic flag are preserved (required by the impl
    // caches in ImplementationsFactory). The OpenCL/runtime members below keep
    // their default null/zero initializers so device state is re-created lazily.
    MegaKernelFastImpl(const MegaKernelFastImpl& other) : cldnn::primitive_impl(other) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> clone() const override {
        return std::make_unique<MegaKernelFastImpl>(*this);
    }
    bool is_cpu() const override { return false; }
    void save(BinaryOutputBuffer&) const override {}
    void load(BinaryInputBuffer&) override {}
    void init_kernels(const cldnn::kernels_cache&, const cldnn::kernel_impl_params&) override {}
    void set_arguments(cldnn::primitive_inst&) override {}
    void set_arguments(cldnn::primitive_inst&, cldnn::kernel_arguments_data&) override {}
    std::vector<cldnn::BufferDescriptor> get_internal_buffer_descs(const cldnn::kernel_impl_params&) const override {
        return {};
    }

    void ensure_ready(cldnn::primitive_inst& instance) {
        std::lock_guard<std::mutex> g(mu_);
        if (ready_) return;

        auto& eng = downcast<ocl_engine>(instance.get_network().get_engine());
        ctx_ = eng.get_cl_context().get();
        dev_ = eng.get_cl_device().get();

        cl_platform_id platform = nullptr;
        clGetDeviceInfo(dev_, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
        usmAlloc_   = reinterpret_cast<clDeviceMemAllocINTEL_fn>(
            clGetExtensionFunctionAddressForPlatform(platform, "clDeviceMemAllocINTEL"));
        usmFree_    = reinterpret_cast<clMemFreeINTEL_fn>(
            clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL"));
        usmMemcpy_  = reinterpret_cast<clEnqueueMemcpyINTEL_fn>(
            clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemcpyINTEL"));
        usmMemFill_ = reinterpret_cast<clEnqueueMemFillINTEL_fn>(
            clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemFillINTEL"));
        OPENVINO_ASSERT(usmAlloc_ && usmFree_ && usmMemcpy_ && usmMemFill_,
                        "[MegaKernel] Intel USM extension functions are unavailable");

        cl_int err;
        prog_ = clCreateProgramWithSource(ctx_, 1, &kKernelSrc, nullptr, &err);
        OPENVINO_ASSERT(err == CL_SUCCESS, "[MegaKernel] clCreateProgramWithSource: ", err);
        err = clBuildProgram(prog_, 1, &dev_, "-cl-std=CL2.0", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t n = 0;
            clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
            std::vector<char> log(n);
            clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
            OPENVINO_THROW("[MegaKernel] Build failed:\n",
                           std::string(log.begin(), log.end()));
        }

        kTask_ = clCreateKernel(prog_, "mk_task", &err);
        OPENVINO_ASSERT(err == CL_SUCCESS, "[MegaKernel] clCreateKernel(mk_task): ", err);

        auto ualloc = [&](size_t bytes) -> void* {
            cl_int st = CL_SUCCESS;
            void* p = usmAlloc_(ctx_, dev_, nullptr, bytes, 0, &st);
            OPENVINO_ASSERT(st == CL_SUCCESS && p, "[MegaKernel] USM device alloc: ", st);
            return p;
        };
        // Per-token scratch (reused across tokens; tokens are serialised).
        mQb_ = ualloc(QDIM   * 4);
        mKb_ = ualloc(KVDIM  * 4);
        mVb_ = ualloc(KVDIM  * 4);
        mGb_ = ualloc(IM_DIM * 2);
        mXn_ = ualloc(QDIM   * 2);
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
        push(0, 0, 0);                                  // embedding
        for (int L = 0; L < NUM_L; L++) {
            for (int i = 0; i < NT_A;  i++) push(1, L, i);   // Stage A  (QKV)
            for (int i = 0; i < NT_BC; i++) push(2, L, i);   // Stage BC (attention)
            for (int i = 0; i < NT_D;  i++) push(3, L, i);   // Stage D  (o-proj)
            for (int i = 0; i < NT_E;  i++) push(4, L, i);   // Stage E  (gate/up)
            for (int i = 0; i < NT_F;  i++) push(5, L, i);   // Stage F  (down)
        }
        auto& strm = instance.get_network().get_stream();
        cl_command_queue q = downcast<ocl_stream>(strm).get_cl_queue().get();
        err = HostInitalizeTaskSystem(taskManager_, queue, dev_, ctx_, q);
        OPENVINO_ASSERT(err == CL_SUCCESS, "[MegaKernel] task-system initialization failed: ", err);

        // TaskManager descriptor consumed by the kernel as a __constant buffer.
        mTaskMgr_ = clCreateBuffer(ctx_,
                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(taskManager_),
                       &taskManager_,
                       &err);
        OPENVINO_ASSERT(err == CL_SUCCESS, "[MegaKernel] clCreateBuffer(taskManager): ", err);

        OPENVINO_ASSERT(clSetKernelArg(kTask_, 0, sizeof(cl_mem), &mTaskMgr_) == CL_SUCCESS,
                        "[MegaKernel] set taskManager arg failed");
        // The tasks reach every data buffer through USM pointers held in the
        // context, so allow the kernel to indirectly access any USM allocation
        // (device / host / shared) regardless of how OpenVINO allocated the weights.
        cl_bool enable = CL_TRUE;
        clSetKernelExecInfo(kTask_, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
                            sizeof(enable), &enable);
        clSetKernelExecInfo(kTask_, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
                            sizeof(enable), &enable);
        clSetKernelExecInfo(kTask_, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
                            sizeof(enable), &enable);
        ready_ = true;
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events,
                              cldnn::primitive_inst& instance) override {
        ensure_ready(instance);

        auto& strm = instance.get_network().get_stream();
        auto& ocls = downcast<ocl_stream>(strm);
        cl_command_queue q = ocls.get_cl_queue().get();

        for (auto& e : events) strm.wait_for_events({e});   // inputs ready before we read them

        // Resolve the raw USM device pointers for every model input/output. The
        // task-system tasks dereference these directly out of the context struct,
        // which requires genuine USM device allocations (asserted below).
        auto usm_raw = [](cldnn::memory& m, const char* name) -> void* {
            auto at = m.get_allocation_type();
            bool usm = at == cldnn::allocation_type::usm_device ||
                       at == cldnn::allocation_type::usm_host ||
                       at == cldnn::allocation_type::usm_shared;
            OPENVINO_ASSERT(usm, "[MegaKernel] input/output '", name,
                            "' must be a USM allocation for the task-system path");
            return m.buffer_ptr();
        };

        MonoCtxH ctx{};
        ctx.hs = usm_raw(instance.input_memory(0),  "hidden_states");
        ctx.h  = usm_raw(instance.output_memory(0), "hidden_states_out");
        ctx.qw = usm_raw(instance.input_memory(5),  "q_proj_w");
        ctx.kw = usm_raw(instance.input_memory(6),  "k_proj_w");
        ctx.vw = usm_raw(instance.input_memory(7),  "v_proj_w");
        ctx.ow = usm_raw(instance.input_memory(8),  "o_proj_w");
        ctx.gw = usm_raw(instance.input_memory(9),  "gate_proj_w");
        ctx.uw = usm_raw(instance.input_memory(10), "up_proj_w");
        ctx.dw = usm_raw(instance.input_memory(11), "down_proj_w");
        ctx.wn = usm_raw(instance.input_memory(12), "input_ln_w");
        ctx.pn = usm_raw(instance.input_memory(13), "post_attn_ln_w");
        ctx.qn = usm_raw(instance.input_memory(14), "q_norm_w");
        ctx.kn = usm_raw(instance.input_memory(15), "k_norm_w");
        ctx.rf = usm_raw(instance.input_memory(16), "rope_inv_freq");
        ctx.qb = mQb_; ctx.kb = mKb_; ctx.vb = mVb_;
        ctx.xn = mXn_; ctx.gbuf = mGb_;
        ctx.kc = mKC_; ctx.vc = mVC_;
        ctx.sync = mSync_;
        ctx.CS = (unsigned)MAX_SEQ;

        // Number of new tokens this step (dim 1 of hidden_states).
        auto hs_ps = instance.input_memory(0).get_layout().get<ov::PartialShape>();
        uint S_new = (uint)hs_ps[1].get_length();

        // Sequence position derived solely from position_ids (input 1), exactly
        // like the original monokernel — the MegaKernel is stateless w.r.t. it.
        static const bool pos_read = [] {
            const char* v = std::getenv("OV_MEGAKERNEL_POSREAD");
            return !(v && v[0] == '0');
        }();
        static thread_local uint acc_len = 0;   // fallback accumulator (diagnostic path)
        uint S_past;
        if (pos_read) {
            const auto& pos_mem = instance.input_memory(1);
            const auto pos_dt = pos_mem.get_layout().data_type;
            int64_t pos0 = -1;
            if (pos_dt == ov::element::i64) {
                pos_mem.copy_to(strm, &pos0, 0, 0, sizeof(int64_t), true);
            } else if (pos_dt == ov::element::i32) {
                int32_t p32 = -1;
                pos_mem.copy_to(strm, &p32, 0, 0, sizeof(int32_t), true);
                pos0 = (int64_t)p32;
            }
            OPENVINO_ASSERT(pos0 >= 0,
                            "[MegaKernel] position_ids (input 1) must be i32/i64 and >= 0; "
                            "the MegaKernel derives its sequence position solely from it.");
            S_past = (uint)pos0;
        } else {
            S_past = (S_new > 1) ? 0u : acc_len;
        }
        acc_len = S_past + S_new;

        // Co-resident worker count (see MONO_WG note). Tunable via env.
        static const int workers = [] {
            const char* v = std::getenv("OV_MEGAKERNEL_MONO_WG");
            int w = v ? atoi(v) : MONO_WG;
            if (w < 1) w = 1;
            if (w > 160) w = 160;
            return w;
        }();

        // ===== Task-system path: the whole 28-layer model as one queue of tasks,
        // launched once per new token. The in-order queue serialises tokens so
        // token t+1 sees the KV cache written by token t. For each token we refresh
        // the shared context (pos / token offset) and reset the sync counters, then
        // launch the worker pool which drains the queue.
        const char pat = 0;
        for (uint t = 0; t < S_new; t++) {
            ctx.pos     = (int)(S_past + t);
            ctx.tok_off = t * (unsigned)H_DIM;
            OPENVINO_ASSERT(usmMemcpy_(q, CL_TRUE, mCtx_, &ctx, sizeof(ctx), 0, nullptr, nullptr) == CL_SUCCESS,
                            "[MegaKernel] context update failed");
            // Zero the stage counters and the FIFO cursor before the workers start.
            usmMemFill_(q, mSync_,    &pat, 1, SYNC_N * sizeof(int), 0, nullptr, nullptr);
            size_t g = (size_t)workers * MONO_LWS, l = (size_t)MONO_LWS;
            cl_int r = clEnqueueNDRangeKernel(q, kTask_, 1, nullptr, &g, &l, 0, nullptr, nullptr);
            OPENVINO_ASSERT(r == CL_SUCCESS, "[MegaKernel] enqueue: ", r);
        }

        cl_event marker;
        clEnqueueMarkerWithWaitList(q, 0, nullptr, &marker);
        return std::make_shared<ocl_event>(cl::Event(marker, false), 0ULL);
    }

private:
    std::mutex mu_;
    bool ready_ = false;
    cl_context    ctx_  = nullptr;
    cl_device_id  dev_  = nullptr;
    cl_program    prog_ = nullptr;
    cl_kernel     kTask_ = nullptr;      // task-system worker kernel (whole model)
    // Intel USM extension entry points (resolved in ensure_ready).
    clDeviceMemAllocINTEL_fn  usmAlloc_   = nullptr;
    clMemFreeINTEL_fn         usmFree_    = nullptr;
    clEnqueueMemcpyINTEL_fn   usmMemcpy_  = nullptr;
    clEnqueueMemFillINTEL_fn  usmMemFill_ = nullptr;
    // USM device allocations: per-token scratch, KV cache, sync counters, context.
    void* mQb_=nullptr; void* mKb_=nullptr; void* mVb_=nullptr; void* mGb_=nullptr; void* mXn_=nullptr;
    void* mKC_=nullptr; void* mVC_=nullptr;
    void* mSync_=nullptr; void* mCtx_=nullptr;
    // Task-system state: work queue, FIFO cursor and the __constant descriptor.
    TaskManager taskManager_{};
    cl_mem mTaskMgr_=nullptr;
};

}  // namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
std::unique_ptr<cldnn::primitive_impl> MegaKernelImpl::create_impl(
        const cldnn::program_node& node, const RuntimeParams& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::megakernel>());
    return std::make_unique<MegaKernelFastImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::megakernel)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MegaKernelFastImpl)
