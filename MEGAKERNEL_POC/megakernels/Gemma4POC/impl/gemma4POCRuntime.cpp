#include "gemma4POCRuntime.h"

#include <CL/cl_ext.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "gemma4FfnKernels.h"
#include "gemma4PrefillKernels.h"

namespace mk {
namespace {

#define CL_OK(expr, what)                                                  \
    do {                                                                   \
        cl_int _e = (expr);                                                \
        if (_e != CL_SUCCESS) {                                            \
            std::fprintf(stderr, "[Gemma4] %s failed: %d\n", what, _e);    \
            return _e;                                                     \
        }                                                                  \
    } while (0)

constexpr int SG = 16;    // one output row per sub-group
constexpr int LWS = 256;  // 16 sub-groups per work-group

// lm_head is u8 with a per-row scale and zero point (it is tied to the int8
// embedding table), so the whole row is a single quantisation group:
//     y[n] = scale[n] * ( dot(q[n,:], x) - zp[n] * sum(x) )
// sum(x) is computed once per token, which keeps the inner loop a plain
// multiply-accumulate over bytes with no per-element subtract.
const char* kSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

__kernel void sum_x(__global const half* x, __global float* sumx, int tokens) {
    int t = get_global_id(0);
    if (t >= tokens) return;
    __global const half* xt = x + (long)t * K_;
    float s = 0.0f;
    for (int i = 0; i < K_; ++i) s += (float)xt[i];
    sumx[t] = s;
}

__attribute__((intel_reqd_sub_group_size(SG_)))
__kernel __attribute__((reqd_work_group_size(LWS_, 1, 1)))
void lm_head_gemv(__global const half* x,
                  __global const uchar* q,
                  __global const uchar* zp,
                  __global const half* sc,
                  __global const float* sumx,
                  __global float* out,
                  int tokens) {
    const int lid = get_local_id(0);
    const int sg_id = lid / SG_;
    const int lane = lid % SG_;
    const int nsg = LWS_ / SG_;

    for (int t = 0; t < tokens; ++t) {
        __global const half* xt = x + (long)t * K_;
        const float sx = sumx[t];

        for (long row = (long)get_group_id(0) * nsg + sg_id;
             row < N_;
             row += (long)get_num_groups(0) * nsg) {

            __global const uchar* qr = q + row * (long)K_;
            float acc = 0.0f;

            // 16 lanes x uchar4 == 64 contiguous bytes per step.
            for (int k = lane * 4; k < K_; k += SG_ * 4) {
                uchar4 v = vload4(0, qr + k);
                half4 h = vload4(0, xt + k);
                acc += (float)v.x * (float)h.x
                     + (float)v.y * (float)h.y
                     + (float)v.z * (float)h.z
                     + (float)v.w * (float)h.w;
            }
            acc = sub_group_reduce_add(acc);

            if (lane == 0) {
                float zv = (float)zp[row];
                out[(long)t * N_ + row] = (float)sc[row] * (acc - zv * sx);
            }
        }
    }
}
)CLC";

cl_program build_sources(cl_context ctx, cl_device_id dev, std::initializer_list<const char*> srcs, const char* opts,
                         cl_int* err) {
    std::vector<const char*> v(srcs);
    cl_program prog = clCreateProgramWithSource(ctx, static_cast<cl_uint>(v.size()), v.data(), nullptr, err);
    if (*err != CL_SUCCESS)
        return nullptr;
    *err = clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
    size_t n = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    std::vector<char> log(n + 1, 0);
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    if (*err != CL_SUCCESS || std::getenv("OV_MEGAKERNEL_DUMP"))
        std::fprintf(stderr, "[Gemma4] build %s:\n%s\n", *err == CL_SUCCESS ? "log" : "FAILED", log.data());
    if (*err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return nullptr;
    }
    return prog;
}

cl_program build(cl_context ctx, cl_device_id dev, const char* src, const char* opts, cl_int* err) {
    return build_sources(ctx, dev, {src}, opts, err);
}

// All 30 kFfn instances share one program per context; pf_mb > 0 selects the
// prefill program built with that GEMM tile height.
cl_program ffn_program(cl_context ctx, cl_device_id dev, cl_int* err, int pf_mb = 0) {
    static std::mutex mu;
    static std::map<std::pair<cl_context, int>, cl_program> cache;
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find({ctx, pf_mb});
    if (it != cache.end()) {
        *err = CL_SUCCESS;
        return it->second;
    }
    char opts[512];
    // OV_MEGAKERNEL_CL_OPTS: extra -D overrides for tuning experiments.
    const char* extra = std::getenv("OV_MEGAKERNEL_CL_OPTS");
    std::snprintf(opts, sizeof(opts), "-DH_=%d -DI_=%d -DIE_=%d -DE_=%d -DTOPK_=%d -DEPS_=1e-6f %s %s",
                  kHidden, kInter, kExpertInter, kExperts, kTopK, kFfnBuildOpts, extra ? extra : "");
    std::string o = opts;
    if (pf_mb)
        o += " -DPF_MB=" + std::to_string(pf_mb) + " -DPF_NSG=" + std::to_string(kPfNsg) + " " + kPrefillBuildOpts;
    cl_program p = pf_mb ? build_sources(ctx, dev, {kFfnSource, kPrefillSource}, o.c_str(), err)
                         : build(ctx, dev, kFfnSource, o.c_str(), err);
    if (p)
        cache.emplace(std::make_pair(ctx, pf_mb), p);
    return p;
}

// Multi-token intermediates, shared by every MegaKernel of a context: the
// layers run one after another on the same in-order queue.
enum PfBuf {
    kPfAT,    // tiled f16 GEMM input (q|k|v or o_proj)
    kPfRaw,   // f32 [T, R] q|k|v projections
    kPfO,     // f32 [T, H] o_proj
    kPfH1,    // f32 [T, H]
    kPfXD,    // tiled f16 [T, H] dense MLP input
    kPfXR,    // tiled f16 [T, H] router input
    kPfX2,    // f16 [T, H] MoE input
    kPfLog,   // f32 [KS, T, E] router partials
    kPfTkI,   // i32 [T, TOPK]
    kPfTkW,   // f32 [T, TOPK]
    kPfAct,   // tiled f16 [T, I]
    kPfMD,    // f32 [T, H] dense MLP output
    kPfHdr,   // i32 [1] MoE tile count
    kPfTExp,  // i32 [tiles]
    kPfPair,  // i32 [tiles * MT]
    kPfXE,    // tiled f16 [tiles * MT, H]
    kPfAE,    // tiled f16 [tiles * MT, IE]
    kPfYS,    // f16 [T * TOPK, H] weighted expert outputs
    kPfBufs
};
struct PfScratch {
    void* buf[kPfBufs] = {};
    size_t cap[kPfBufs] = {};
    int refs = 0;
};
std::mutex g_pf_mu;
std::map<cl_context, PfScratch> g_pf;

enum PfKernel { kPfAttnRms, kPfGemm, kPfTile, kPfFfnIn, kPfGemmF16w, kPfTopk, kPfGemmGelu, kPfRoute, kPfMoeDown,
                kPfFfnOut, kPfGather, kPfKernels };
const char* const kPfKernelNames[kPfKernels] = {"pf_attn_rms", "pf_gemm",     "pf_tile",      "pf_ffn_in",
                                                "pf_gemm_f16w", "pf_topk",     "pf_gemm_gelu", "pf_moe_route",
                                                "pf_gemm_moe_down", "pf_ffn_out", "pf_gather"};

// Calls of at least this many tokens take the batched path (OV_MEGAKERNEL_PF_MIN).
int prefill_min_tokens() {
    static const int v = [] {
        const char* e = std::getenv("OV_MEGAKERNEL_PF_MIN");
        return e ? std::atoi(e) : 2;
    }();
    return v;
}

size_t round_up(size_t v, size_t m) {
    return (v + m - 1) / m * m;
}

// Scratch layout of one kFfn token (floats unless noted).
enum : size_t {
    kScO = 0,
    kScH1 = kScO + kHidden,
    kScX2 = kScH1 + kHidden,
    kScAct = kScX2 + kHidden,
    kScLogits = kScAct + kInter,
    kScMoeAct = kScLogits + kExperts,
    kScM = kScMoeAct + kTopK * kExpertInter,
    kScE = kScM + kHidden,
    kScTkIdx = kScE + kHidden,  // i32
    kScTkW = kScTkIdx + kTopK,
    kScSize = kScTkW + kTopK,
};

enum FfnKernel { kOProj, kIn, kMid, kMoeDown, kOut };
constexpr int kPrefillSyncTokens = 64;
const char* const kFfnKernelNames[kFfnKernels] = {"ffn_oproj", "ffn_in", "ffn_mid", "ffn_moe_down", "ffn_out"};

}  // namespace

TErrorcode Gemma4POCRuntime::Init(const IConstantParams* weights, const IPlatformParams* platform) {
    w_ = *static_cast<const Gemma4ConstantParams*>(weights);
    const auto* pp = static_cast<const Gemma4PlatformParams*>(platform);
    ctx_ = pp->context;
    dev_ = pp->deviceId;
    stream_ = pp->stream;

    cl_platform_id plat = nullptr;
    clGetDeviceInfo(dev_, CL_DEVICE_PLATFORM, sizeof(plat), &plat, nullptr);
    usmAlloc_ = reinterpret_cast<clDeviceMemAllocINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(plat, "clDeviceMemAllocINTEL"));
    usmFree_ = reinterpret_cast<clMemFreeINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(plat, "clMemFreeINTEL"));
    setUsmArg_ = reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(plat, "clSetKernelArgMemPointerINTEL"));
    usmBlockingFree_ = reinterpret_cast<clMemBlockingFreeINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(plat, "clMemBlockingFreeINTEL"));
    if (!usmAlloc_ || !usmFree_ || !setUsmArg_) {
        std::fprintf(stderr, "[Gemma4] Intel USM extensions unavailable\n");
        return CL_INVALID_PLATFORM;
    }
    if (w_.kind == kAttnIn)
        return InitAttn();
    if (w_.kind == kMaskFull || w_.kind == kMaskSliding)
        return InitMask();
    return w_.kind == kFfn ? InitFfn() : InitLmHead();
}

TErrorcode Gemma4POCRuntime::InitFfn() {
    cl_int err = CL_SUCCESS;
    cl_program prog = ffn_program(ctx_, dev_, &err);
    CL_OK(err, "build ffn program");
    for (int i = 0; i < kFfnKernels; ++i) {
        kFfn_[i] = clCreateKernel(prog, kFfnKernelNames[i], &err);
        CL_OK(err, kFfnKernelNames[i]);
    }

    scratch_ = usmAlloc_(ctx_, dev_, nullptr, kScSize * sizeof(float), 0, &err);
    CL_OK(err, "scratch alloc");
    auto* sc = static_cast<float*>(scratch_);

    ffn_global_sizes(kHidden, kInter, kExpertInter, kExperts, kTopK, ffnGws_);

    if (std::getenv("OV_MEGAKERNEL_DUMP")) {
        for (cl_kernel k : kFfn_) {
            cl_ulong spill = 0, slm = 0;
            char name[64] = {0};
            clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(name) - 1, name, nullptr);
            clGetKernelWorkGroupInfo(k, dev_, 0x4109 /*CL_KERNEL_SPILL_MEM_SIZE_INTEL*/, sizeof(spill), &spill, nullptr);
            clGetKernelWorkGroupInfo(k, dev_, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(slm), &slm, nullptr);
            std::fprintf(stderr, "[Gemma4] kernel %s spill=%llu slm=%llu\n", name, (unsigned long long)spill,
                         (unsigned long long)slm);
        }
    }

    auto setp = [&](cl_kernel k, cl_uint i, const void* p) { setUsmArg_(k, i, p); };
    void* const* f = w_.ffn;
    // ffn_oproj: args 0 (attention output) and 2 (its width) are per token.
    setp(kFfn_[kOProj], 1, f[kFfnOProj]);
    setp(kFfn_[kOProj], 3, sc + kScO);

    // ffn_in: arg 1 (residual) is per token.
    setp(kFfn_[kIn], 0, sc + kScO);
    cl_uint a = 2;
    for (int port : {kFfnWPostAttn, kFfnWPreFf, kFfnWRouter, kFfnWPreFf2, kFfnGate, kFfnUp, kFfnRouterW})
        setp(kFfn_[kIn], a++, f[port]);
    for (size_t off : {kScH1, kScX2, kScAct, kScLogits})
        setp(kFfn_[kIn], a++, sc + off);

    a = 0;
    for (size_t off : {kScX2, kScLogits, kScAct})
        setp(kFfn_[kMid], a++, sc + off);
    for (int port : {kFfnEGate, kFfnEUp, kFfnDown})
        setp(kFfn_[kMid], a++, f[port]);
    for (size_t off : {kScMoeAct, kScM, kScTkIdx, kScTkW})
        setp(kFfn_[kMid], a++, sc + off);

    a = 0;
    for (size_t off : {kScMoeAct, kScTkIdx, kScTkW})
        setp(kFfn_[kMoeDown], a++, sc + off);
    setp(kFfn_[kMoeDown], a++, f[kFfnEDown]);
    setp(kFfn_[kMoeDown], a++, sc + kScE);

    a = 0;
    for (size_t off : {kScH1, kScM, kScE})
        setp(kFfn_[kOut], a++, sc + off);
    for (int port : {kFfnWPostFf1, kFfnWPostFf2, kFfnWPostFf, kFfnLayerScalar})
        setp(kFfn_[kOut], a++, f[port]);
    // arg 7 (out) is per token.
    return InitPrefill();
}

TErrorcode Gemma4POCRuntime::InitPrefill() {
    for (int v = 0; v < 2; ++v) {
        cl_int err = CL_SUCCESS;
        cl_program prog = ffn_program(ctx_, dev_, &err, v ? kPfMbSmall : kPfMb);
        CL_OK(err, "build prefill program");
        for (int i = 0; i < kPfKernels; ++i) {
            kPf_[v][i] = clCreateKernel(prog, kPfKernelNames[i], &err);
            CL_OK(err, kPfKernelNames[i]);
            if (std::getenv("OV_MEGAKERNEL_DUMP")) {
                cl_ulong spill = 0;
                clGetKernelWorkGroupInfo(kPf_[v][i], dev_, 0x4109 /*CL_KERNEL_SPILL_MEM_SIZE_INTEL*/, sizeof(spill),
                                         &spill, nullptr);
                std::fprintf(stderr, "[Gemma4] kernel %s/%d spill=%llu\n", kPfKernelNames[i], v,
                             (unsigned long long)spill);
            }
        }
    }
    std::lock_guard<std::mutex> g(g_pf_mu);
    ++g_pf[ctx_].refs;
    pfShared_ = true;
    return CL_SUCCESS;
}

void* Gemma4POCRuntime::pf_buf(int id, size_t bytes) {
    std::lock_guard<std::mutex> g(g_pf_mu);
    PfScratch& s = g_pf[ctx_];
    if (s.cap[id] < bytes) {
        // Earlier layers of this call may still use the old block.
        if (s.buf[id])
            usmBlockingFree_(ctx_, s.buf[id]);
        cl_int err = CL_SUCCESS;
        const size_t cap = round_up(bytes + bytes / 4, 4096);
        s.buf[id] = usmAlloc_(ctx_, dev_, nullptr, cap, 0, &err);
        s.cap[id] = err == CL_SUCCESS && s.buf[id] ? cap : 0;
        if (!s.cap[id])
            s.buf[id] = nullptr;
    }
    return s.buf[id];
}

// Kernel k of the tile variant `v` in scope (0: kPfMb, 1: kPfMbSmall).
#define PF_ARG(k, i, x) CL_OK(clSetKernelArg(kPf_[v][k], i, sizeof(x), &(x)), kPfKernelNames[k])
#define PF_PTR(k, i, p) CL_OK(setUsmArg_(kPf_[v][k], i, p), kPfKernelNames[k])
#define PF_RUN(k, dims, gws, lws) \
    CL_OK(clEnqueueNDRangeKernel(stream_, kPf_[v][k], dims, nullptr, gws, lws, 0, nullptr, nullptr), kPfKernelNames[k])

// Variant for the dense GEMMs of a T-token call, and for its MoE.
static int pf_dense_variant(int T) {
    static const int lim = [] {
        const char* e = std::getenv("OV_MEGAKERNEL_PF_DENSE_SMALL");
        return e ? std::atoi(e) : 32;
    }();
    return T <= lim ? 1 : 0;
}
static int pf_moe_variant(int T) {
    // Few rows per expert: 8-row tiles waste far less XMX work than 64-row ones.
    static const int lim = [] {
        const char* e = std::getenv("OV_MEGAKERNEL_PF_MOE_SMALL");
        return e ? std::atoi(e) : 256;
    }();
    return T <= lim ? 1 : 0;
}
static int pf_mt(int v) {
    return 8 * (v ? kPfMbSmall : kPfMb);
}

TErrorcode Gemma4POCRuntime::PrefillAttn(const Gemma4RuntimeParams& p) {
    const int T = p.tokens, R = w_.attn_rows, H = kHidden;
    const int v = pf_dense_variant(T);
    const size_t mtiles = (T + pf_mt(v) - 1) / pf_mt(v);
    void* at = pf_buf(kPfAT, static_cast<size_t>(T) * H * sizeof(uint16_t));
    void* raw = pf_buf(kPfRaw, static_cast<size_t>(T) * R * sizeof(float));
    if (!at || !raw)
        return CL_OUT_OF_RESOURCES;
    void* const* a = w_.attn;

    PF_PTR(kPfAttnRms, 0, p.in);
    PF_PTR(kPfAttnRms, 1, a[kAttnWIn]);
    PF_PTR(kPfAttnRms, 2, at);
    const size_t rms_gws = kFfnLws * T, lws = kFfnLws;
    PF_RUN(kPfAttnRms, 1, &rms_gws, &lws);

    const int nrb = R / 16;
    PF_PTR(kPfGemm, 0, at);
    PF_ARG(kPfGemm, 1, H);
    PF_PTR(kPfGemm, 2, a[kAttnQkv]);
    PF_ARG(kPfGemm, 3, nrb);
    PF_ARG(kPfGemm, 4, T);
    PF_PTR(kPfGemm, 5, raw);
    PF_ARG(kPfGemm, 6, R);
    const size_t g_gws[2] = {round_up(nrb / 2, kPfNsg) / kPfNsg * kPfLws, mtiles};
    const size_t g_lws[2] = {kPfLws, 1};
    PF_RUN(kPfGemm, 2, g_gws, g_lws);

    const int t0 = 0;
    setUsmArg_(kAttn_[1], 0, raw);
    setUsmArg_(kAttn_[1], 4, p.in2);
    setUsmArg_(kAttn_[1], 6, p.out);
    CL_OK(clSetKernelArg(kAttn_[1], 7, sizeof(int), &t0), "set t0");
    CL_OK(clSetKernelArg(kAttn_[1], 8, sizeof(int), &p.s), "set S");
    CL_OK(clSetKernelArg(kAttn_[1], 9, sizeof(int), &R), "set rstride");
    const size_t post_gws[2] = {attnGws_[1], static_cast<size_t>(T)};
    const size_t post_lws[2] = {kFfnLws, 1};
    CL_OK(clEnqueueNDRangeKernel(stream_, kAttn_[1], 2, nullptr, post_gws, post_lws, 0, nullptr, nullptr),
          "attn_post");
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::PrefillFfn(const Gemma4RuntimeParams& p) {
    const int T = p.tokens, H = kHidden, KA = p.ka, I = kInter, IE = kExpertInter, E = kExperts;
    const int dv = pf_dense_variant(T), mv = pf_moe_variant(T);
    const size_t mtiles = (T + pf_mt(dv) - 1) / pf_mt(dv);
    const size_t emt = pf_mt(mv);
    const size_t etiles = (static_cast<size_t>(T) * kTopK + emt - 1) / emt + E;  // bound on MoE tiles
    const size_t t = T;
    void* at = pf_buf(kPfAT, t * std::max(KA, H) * 2);
    void* o = pf_buf(kPfO, t * H * 4);
    void* h1 = pf_buf(kPfH1, t * H * 4);
    void* xd = pf_buf(kPfXD, t * H * 2);
    void* xr = pf_buf(kPfXR, t * H * 2);
    void* x2 = pf_buf(kPfX2, t * H * 2);
    void* lg = pf_buf(kPfLog, kPfRouterKs * t * E * 4);
    void* tki = pf_buf(kPfTkI, t * kTopK * 4);
    void* tkw = pf_buf(kPfTkW, t * kTopK * 4);
    void* act = pf_buf(kPfAct, t * I * 2);
    void* md = pf_buf(kPfMD, t * H * 4);
    void* hdr = pf_buf(kPfHdr, 64);
    void* texp = pf_buf(kPfTExp, etiles * 4);
    void* pair = pf_buf(kPfPair, etiles * emt * 4);
    void* xe = pf_buf(kPfXE, etiles * emt * H * 2);
    void* ae = pf_buf(kPfAE, etiles * emt * IE * 2);
    void* ys = pf_buf(kPfYS, t * kTopK * H * 2);
    for (void* b : {at, o, h1, xd, xr, x2, lg, tki, tkw, act, md, hdr, texp, pair, xe, ae, ys})
        if (!b)
            return CL_OUT_OF_RESOURCES;
    void* const* f = w_.ffn;
    void* const null = nullptr;
    const size_t lws = kFfnLws, tok_gws = kFfnLws * T;
    const size_t g_lws[2] = {kPfLws, 1};
    const size_t tile_lws[2] = {16, 16};

    auto gemm = [&](int v, void* a, int K, void* w, int N, void* out) -> TErrorcode {
        const int nrb = N / 16;
        PF_PTR(kPfGemm, 0, a);
        PF_ARG(kPfGemm, 1, K);
        PF_PTR(kPfGemm, 2, w);
        PF_ARG(kPfGemm, 3, nrb);
        PF_ARG(kPfGemm, 4, T);
        PF_PTR(kPfGemm, 5, out);
        PF_ARG(kPfGemm, 6, N);
        const size_t gws[2] = {round_up(nrb / 2, kPfNsg) / kPfNsg * kPfLws, mtiles};
        PF_RUN(kPfGemm, 2, gws, g_lws);
        return CL_SUCCESS;
    };
    auto gelu = [&](int v, void* a, int K, void* wg, void* wu, int N, void* hd, void* te, size_t tiles,
                    void* out) -> TErrorcode {
        const int nrb = N / 16;
        PF_PTR(kPfGemmGelu, 0, a);
        PF_ARG(kPfGemmGelu, 1, K);
        PF_PTR(kPfGemmGelu, 2, wg);
        PF_PTR(kPfGemmGelu, 3, wu);
        PF_ARG(kPfGemmGelu, 4, nrb);
        PF_ARG(kPfGemmGelu, 5, T);
        PF_PTR(kPfGemmGelu, 6, hd);
        PF_PTR(kPfGemmGelu, 7, te);
        PF_PTR(kPfGemmGelu, 8, out);
        const size_t gws[2] = {round_up(nrb, kPfNsg) / kPfNsg * kPfLws, tiles};
        PF_RUN(kPfGemmGelu, 2, gws, g_lws);
        return CL_SUCCESS;
    };
    auto tile = [&](int v, void* src, int K, void* out) -> TErrorcode {
        PF_PTR(kPfTile, 0, src);
        PF_ARG(kPfTile, 1, K);
        PF_ARG(kPfTile, 2, T);
        PF_PTR(kPfTile, 3, out);
        const size_t gws[2] = {round_up(K / 16, 16), round_up(t, 16)};
        PF_RUN(kPfTile, 2, gws, tile_lws);
        return CL_SUCCESS;
    };
#define PF_DO(x)                     \
    do {                             \
        const TErrorcode e_ = (x);   \
        if (e_ != CL_SUCCESS)        \
            return e_;               \
    } while (0)

    // o = o_proj(attn)
    PF_DO(tile(dv, p.in, KA, at));
    PF_DO(gemm(dv, at, KA, f[kFfnOProj], H, o));

    int v = 0;
    // h1, the three normalised activations
    {
        cl_uint a = 0;
        for (void* x : {o, p.in2, f[kFfnWPostAttn], f[kFfnWPreFf], f[kFfnWRouter], f[kFfnWPreFf2], h1, xd, xr, x2})
            PF_PTR(kPfFfnIn, a++, x);
        PF_RUN(kPfFfnIn, 1, &tok_gws, &lws);
    }

    // router logits (K-split partials), top-k
    {
        PF_PTR(kPfGemmF16w, 0, xr);
        PF_ARG(kPfGemmF16w, 1, H);
        PF_PTR(kPfGemmF16w, 2, f[kFfnRouterW]);
        PF_ARG(kPfGemmF16w, 3, E);
        PF_ARG(kPfGemmF16w, 4, T);
        PF_PTR(kPfGemmF16w, 5, lg);
        const size_t gws[2] = {kPfRouterKs * kPfLws, (t + kPfMt - 1) / kPfMt};
        PF_RUN(kPfGemmF16w, 2, gws, g_lws);
        const int ks = kPfRouterKs;
        PF_PTR(kPfTopk, 0, lg);
        PF_ARG(kPfTopk, 1, T);
        PF_ARG(kPfTopk, 2, ks);
        PF_PTR(kPfTopk, 3, tki);
        PF_PTR(kPfTopk, 4, tkw);
        const size_t tk_gws = round_up(t, 16) / 16 * kFfnLws;
        PF_RUN(kPfTopk, 1, &tk_gws, &lws);
    }

    // dense MLP
    PF_DO(gelu(dv, xd, H, f[kFfnGate], f[kFfnUp], I, null, null, mtiles, act));
    PF_DO(gemm(dv, act, I, f[kFfnDown], H, md));

    // MoE: bucket pairs per expert, gather, grouped gate/up, grouped down + scatter
    v = mv;
    {
        PF_PTR(kPfRoute, 0, tki);
        PF_ARG(kPfRoute, 1, T);
        PF_PTR(kPfRoute, 2, hdr);
        PF_PTR(kPfRoute, 3, texp);
        PF_PTR(kPfRoute, 4, pair);
        const size_t r_gws = 1024;
        PF_RUN(kPfRoute, 1, &r_gws, &r_gws);
    }
    {
        PF_PTR(kPfGather, 0, x2);
        PF_ARG(kPfGather, 1, H);
        PF_PTR(kPfGather, 2, hdr);
        PF_PTR(kPfGather, 3, pair);
        PF_PTR(kPfGather, 4, xe);
        const size_t lw[2] = {static_cast<size_t>(H) * 2 / 512 * 16, 1};
        const size_t gws[2] = {lw[0], etiles * emt};
        PF_RUN(kPfGather, 2, gws, lw);
    }
    PF_DO(gelu(mv, xe, H, f[kFfnEGate], f[kFfnEUp], IE, hdr, texp, etiles, ae));
    {
        const int nrb = H / 16;
        cl_uint a = 0;
        PF_PTR(kPfMoeDown, a++, ae);
        PF_ARG(kPfMoeDown, a++, IE);
        PF_PTR(kPfMoeDown, a++, f[kFfnEDown]);
        PF_ARG(kPfMoeDown, a++, nrb);
        for (void* x : {hdr, texp, pair, tkw, ys})
            PF_PTR(kPfMoeDown, a++, x);
        const size_t gws[2] = {round_up(nrb / 2, kPfNsg) / kPfNsg * kPfLws, etiles};
        PF_RUN(kPfMoeDown, 2, gws, g_lws);
    }

    // out
    v = 0;
    {
        cl_uint a = 0;
        for (void* x : {h1, md, ys, f[kFfnWPostFf1], f[kFfnWPostFf2], f[kFfnWPostFf], f[kFfnLayerScalar], p.out})
            PF_PTR(kPfFfnOut, a++, x);
        PF_RUN(kPfFfnOut, 1, &tok_gws, &lws);
    }
#undef PF_DO
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::ExecuteFfn(const Gemma4RuntimeParams& p) {
    if (p.tokens >= prefill_min_tokens())
        return PrefillFfn(p);
    const size_t lws = kFfnLws;
    const size_t row = kHidden * sizeof(uint16_t);
    const size_t attn_row = static_cast<size_t>(p.ka) * sizeof(uint16_t);
    CL_OK(clSetKernelArg(kFfn_[kOProj], 2, sizeof(int), &p.ka), "set o_proj width");
    // The block is token-local, so prefill simply repeats the decode launches.
    for (int t = 0; t < p.tokens; ++t) {
        setUsmArg_(kFfn_[kOProj], 0, static_cast<char*>(p.in) + t * attn_row);
        setUsmArg_(kFfn_[kIn], 1, static_cast<char*>(p.in2) + t * row);
        setUsmArg_(kFfn_[kOut], 7, static_cast<char*>(p.out) + t * row);
        for (int k = 0; k < kFfnKernels; ++k)
            CL_OK(clEnqueueNDRangeKernel(stream_, kFfn_[k], 1, nullptr, &ffnGws_[k], &lws, 0, nullptr, nullptr),
                  kFfnKernelNames[k]);
        // A long prefill would otherwise enqueue ~10^6 launches without ever
        // blocking and exhaust the driver's command storage (CL_OUT_OF_RESOURCES
        // at ~5k tokens), so bound the work in flight.
        if (t % kPrefillSyncTokens == kPrefillSyncTokens - 1)
            CL_OK(clFinish(stream_), "prefill throttle");
    }
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::InitLmHead() {
    char opts[512];
    std::snprintf(opts, sizeof(opts),
                  "-DSG_=%d -DLWS_=%d -DN_=%d -DK_=%d -cl-fast-relaxed-math",
                  SG, LWS, w_.N, w_.K);

    cl_int err = CL_SUCCESS;
    prog_ = build(ctx_, dev_, kSource, opts, &err);
    CL_OK(err, "build lm_head program");

    kGemv_ = clCreateKernel(prog_, "lm_head_gemv", &err);
    CL_OK(err, "clCreateKernel(lm_head_gemv)");
    kSumx_ = clCreateKernel(prog_, "sum_x", &err);
    CL_OK(err, "clCreateKernel(sum_x)");
    return CL_SUCCESS;
}

void Gemma4POCRuntime::ensure_scratch(int tokens) {
    if (tokens <= scratch_tokens_ && mSumx_) return;
    if (!usmAlloc_) return;
    if (mSumx_) usmFree_(ctx_, mSumx_);
    cl_int e = CL_SUCCESS;
    mSumx_ = usmAlloc_(ctx_, dev_, nullptr, (size_t)tokens * sizeof(float), 0, &e);
    scratch_tokens_ = tokens;
}

TErrorcode Gemma4POCRuntime::Execute(const IRuntimeParams* io) {
    const auto* p = static_cast<const Gemma4RuntimeParams*>(io);
    if (!setUsmArg_) {
        std::fprintf(stderr, "[Gemma4] runtime not initialised\n");
        return CL_INVALID_KERNEL;
    }
    if (w_.kind == kFfn)
        return kFfn_[kOut] ? ExecuteFfn(*p) : CL_INVALID_KERNEL;
    if (w_.kind == kAttnIn)
        return kAttn_[1] ? ExecuteAttn(*p) : CL_INVALID_KERNEL;
    if (w_.kind == kMaskFull || w_.kind == kMaskSliding)
        return kMask_[1] ? ExecuteMask(*p) : CL_INVALID_KERNEL;
    return ExecuteLmHead(*p);
}

TErrorcode Gemma4POCRuntime::InitMask() {
    cl_int err = CL_SUCCESS;
    cl_program prog = ffn_program(ctx_, dev_, &err);
    CL_OK(err, "build ffn program");
    kMask_[0] = clCreateKernel(prog, "mask_groups", &err);
    CL_OK(err, "mask_groups");
    kMask_[1] = clCreateKernel(prog, "mask_fill", &err);
    CL_OK(err, "mask_fill");
    setUsmArg_(kMask_[0], 1, w_.mask_meta);
    setUsmArg_(kMask_[1], 1, w_.mask_meta);
    const int sliding = w_.kind == kMaskSliding;
    CL_OK(clSetKernelArg(kMask_[1], 5, sizeof(int), &sliding), "set sliding");
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::ExecuteMask(const Gemma4RuntimeParams& p) {
    const int B = p.tokens, L = p.ka, S = p.s;
    if (B <= 0 || L <= 0 || S <= 0 || S > L)
        return CL_INVALID_VALUE;
    if (w_.kind == kMaskSliding) {
        const size_t need = static_cast<size_t>(B) * L;
        if (need > maskGrpCap_) {
            if (scratch_ && usmBlockingFree_)
                usmBlockingFree_(ctx_, scratch_);
            cl_int err = CL_SUCCESS;
            maskGrpCap_ = need * 2;
            scratch_ = usmAlloc_(ctx_, dev_, nullptr, maskGrpCap_ * sizeof(int), 0, &err);
            CL_OK(err, "mask scratch alloc");
        }
        setUsmArg_(kMask_[0], 0, p.in2);
        CL_OK(clSetKernelArg(kMask_[0], 2, sizeof(int), &S), "set S");
        CL_OK(clSetKernelArg(kMask_[0], 3, sizeof(int), &L), "set L");
        setUsmArg_(kMask_[0], 4, scratch_);
        const size_t gws = 256 * static_cast<size_t>(B), lws = 256;
        CL_OK(clEnqueueNDRangeKernel(stream_, kMask_[0], 1, nullptr, &gws, &lws, 0, nullptr, nullptr),
              "mask_groups");
    }
    setUsmArg_(kMask_[1], 0, p.in);
    setUsmArg_(kMask_[1], 2, scratch_);
    CL_OK(clSetKernelArg(kMask_[1], 3, sizeof(int), &S), "set S");
    CL_OK(clSetKernelArg(kMask_[1], 4, sizeof(int), &L), "set L");
    setUsmArg_(kMask_[1], 6, p.out);
    const size_t gws[3] = {(static_cast<size_t>(L) + 63) / 64 * 64, static_cast<size_t>(S), static_cast<size_t>(B)};
    const size_t lws[3] = {64, 1, 1};
    CL_OK(clEnqueueNDRangeKernel(stream_, kMask_[1], 3, nullptr, gws, lws, 0, nullptr, nullptr), "mask_fill");
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::InitAttn() {
    if (w_.attn_rows <= 0 || w_.attn_rows % kAttnQkvRowsPerWg || w_.attn_rows > kAttnMaxRows) {
        std::fprintf(stderr, "[Gemma4] bad q|k|v row count %d\n", w_.attn_rows);
        return CL_INVALID_VALUE;
    }
    cl_int err = CL_SUCCESS;
    cl_program prog = ffn_program(ctx_, dev_, &err);
    CL_OK(err, "build ffn program");
    kAttn_[0] = clCreateKernel(prog, "attn_qkv", &err);
    CL_OK(err, "attn_qkv");
    kAttn_[1] = clCreateKernel(prog, "attn_post", &err);
    CL_OK(err, "attn_post");

    scratch_ = usmAlloc_(ctx_, dev_, nullptr, w_.attn_rows * sizeof(float), 0, &err);
    CL_OK(err, "scratch alloc");
    attnGws_[0] = kFfnLws * (w_.attn_rows / kAttnQkvRowsPerWg);
    attnGws_[1] = kFfnLws * (kAttnMaxHeads / 16);

    void* const* a = w_.attn;
    setUsmArg_(kAttn_[0], 1, a[kAttnWIn]);
    setUsmArg_(kAttn_[0], 2, a[kAttnQkv]);
    setUsmArg_(kAttn_[0], 3, scratch_);
    setUsmArg_(kAttn_[1], 0, scratch_);
    setUsmArg_(kAttn_[1], 1, a[kAttnWQ]);
    setUsmArg_(kAttn_[1], 2, a[kAttnWK]);
    setUsmArg_(kAttn_[1], 3, a[kAttnMeta]);
    setUsmArg_(kAttn_[1], 5, a[kAttnInvFreq]);
    return InitPrefill();
}

TErrorcode Gemma4POCRuntime::ExecuteAttn(const Gemma4RuntimeParams& p) {
    if (p.tokens >= prefill_min_tokens())
        return PrefillAttn(p);
    const size_t lws = kFfnLws;
    const size_t row = kHidden * sizeof(uint16_t);
    const int rstride = 0;
    setUsmArg_(kAttn_[1], 0, scratch_);
    setUsmArg_(kAttn_[1], 4, p.in2);
    setUsmArg_(kAttn_[1], 6, p.out);
    CL_OK(clSetKernelArg(kAttn_[1], 8, sizeof(int), &p.s), "set S");
    CL_OK(clSetKernelArg(kAttn_[1], 9, sizeof(int), &rstride), "set rstride");
    for (int t = 0; t < p.tokens; ++t) {
        setUsmArg_(kAttn_[0], 0, static_cast<char*>(p.in) + t * row);
        CL_OK(clSetKernelArg(kAttn_[1], 7, sizeof(int), &t), "set token");
        for (int k = 0; k < 2; ++k)
            CL_OK(clEnqueueNDRangeKernel(stream_, kAttn_[k], 1, nullptr, &attnGws_[k], &lws, 0, nullptr, nullptr),
                  k ? "attn_post" : "attn_qkv");
        if (t % kPrefillSyncTokens == kPrefillSyncTokens - 1)
            CL_OK(clFinish(stream_), "prefill throttle");
    }
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::ExecuteLmHead(const Gemma4RuntimeParams& pr) {
    const auto* p = &pr;
    if (!kGemv_ || !kSumx_)
        return CL_INVALID_KERNEL;
    ensure_scratch(p->tokens);
    if (!mSumx_) return CL_OUT_OF_RESOURCES;

    auto setUsm = setUsmArg_;

    setUsm(kSumx_, 0, p->in);
    setUsm(kSumx_, 1, mSumx_);
    clSetKernelArg(kSumx_, 2, sizeof(int), &p->tokens);
    size_t s_gws = ((size_t)p->tokens + 15) / 16 * 16;
    size_t s_lws = 16;
    CL_OK(clEnqueueNDRangeKernel(stream_, kSumx_, 1, nullptr, &s_gws, &s_lws, 0, nullptr, nullptr),
          "enqueue sum_x");

    int a = 0;
    setUsm(kGemv_, a++, p->in);
    setUsm(kGemv_, a++, w_.lm_head_q);
    setUsm(kGemv_, a++, w_.lm_head_zp);
    setUsm(kGemv_, a++, w_.lm_head_scale);
    setUsm(kGemv_, a++, mSumx_);
    setUsm(kGemv_, a++, p->out);
    clSetKernelArg(kGemv_, a++, sizeof(int), &p->tokens);

    cl_uint cus = 0;
    clGetDeviceInfo(dev_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
    size_t wgs = cus ? (size_t)cus * 4 : 256;
    size_t gws = wgs * LWS;
    size_t lws = LWS;
    CL_OK(clEnqueueNDRangeKernel(stream_, kGemv_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr),
          "enqueue lm_head_gemv");
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::Destroy() {
    for (cl_kernel* k : {&kGemv_, &kSumx_}) {
        if (*k)
            clReleaseKernel(*k);
        *k = nullptr;
    }
    for (cl_kernel& k : kFfn_) {
        if (k)
            clReleaseKernel(k);
        k = nullptr;
    }
    for (cl_kernel& k : kAttn_) {
        if (k)
            clReleaseKernel(k);
        k = nullptr;
    }
    for (cl_kernel& k : kMask_) {
        if (k)
            clReleaseKernel(k);
        k = nullptr;
    }
    for (auto& ks : kPf_) {
        for (cl_kernel& k : ks) {
            if (k)
                clReleaseKernel(k);
            k = nullptr;
        }
    }
    if (pfShared_) {
        std::lock_guard<std::mutex> g(g_pf_mu);
        PfScratch& s = g_pf[ctx_];
        if (--s.refs == 0) {
            for (void* b : s.buf)
                if (b && usmBlockingFree_)
                    usmBlockingFree_(ctx_, b);
            g_pf.erase(ctx_);
        }
        pfShared_ = false;
    }
    if (prog_) clReleaseProgram(prog_);
    prog_ = nullptr;
    // In-flight kernels may still read the scratch; the blocking free waits for
    // them without touching the queue, which may already be gone.
    for (void** m : {&scratch_, &mSumx_}) {
        if (*m && usmBlockingFree_)
            usmBlockingFree_(ctx_, *m);
        *m = nullptr;
    }
    return CL_SUCCESS;
}

}  // namespace mk
