#include "gemma4ModelRuntime.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>

#include "gemma4FfnKernels.h"
#include "gemma4ModelKernels.h"
#include "gemma4POCRuntime.h"

namespace mk {
namespace {

#define MK_OK(expr, what)                                                       \
    do {                                                                        \
        cl_int _e = (expr);                                                     \
        if (_e != CL_SUCCESS) {                                                 \
            std::fprintf(stderr, "[Gemma4Model] %s failed: %d\n", what, _e);    \
            return _e;                                                          \
        }                                                                       \
    } while (0)

constexpr int kLws = 256;
constexpr int kRMax = kAttnMaxRows;
constexpr int kPhasesPerLayer = 6;  // NPH in gemma4ModelKernels.h

int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : def;
}

size_t round_up(size_t v, size_t m) {
    return (v + m - 1) / m * m;
}

// Device-side table layouts; must match LayerTab / ModelTab in gemma4ModelKernels.h.
struct LayerTabH {
    const void *w_in, *qkv, *gq, *gk, *invf, *oproj, *w_pa, *w_pf, *w_r, *w_pf2, *gate, *up, *down, *rw, *egate,
        *eup, *edown, *w_pf1, *w_pf2p, *w_post, *ls;
    void *kc, *vc;
    int32_t R, HQ, HK, D, RN, q_off, k_off, v_off, half_c, pos_f16, sliding, pad;
};
static_assert(sizeof(LayerTabH) == 23 * 8 + 12 * 4, "LayerTab layout");

struct ModelTabH {
    void *xbuf, *raw, *att, *o, *h1, *gu, *lg, *egu, *m, *e, *part, *sync, *tick, *err;
    const void *w_final, *lm, *lm_s, *lm_z;
    float cap_div, cap_mul, attn_scale, window, causal_off, p_mul, p_add, pad0;
    int32_t pos_stride, am_stride, kv_cap, pad1;
    void* prof;
    void *xqm, *xqd;
};
static_assert(sizeof(ModelTabH) == 21 * 8 + 12 * 4, "ModelTab layout");

enum Scratch {
    kSXbuf, kSRaw, kSAtt, kSO, kSH1, kSGu, kSLg, kSEgu, kSM, kSE, kSPart, kSSync, kSTick, kSErr, kSXqm, kSXqd, kSCount
};

cl_program build_model_program(cl_context ctx, cl_device_id dev, const std::string& opts, cl_int* err) {
    const char* srcs[2] = {kFfnSource, kModelSource};
    cl_program prog = clCreateProgramWithSource(ctx, 2, srcs, nullptr, err);
    if (*err != CL_SUCCESS)
        return nullptr;
    *err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    size_t n = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    std::vector<char> log(n + 1, 0);
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    if (*err != CL_SUCCESS || std::getenv("OV_MEGAKERNEL_DUMP"))
        std::fprintf(stderr, "[Gemma4Model] build %s:\n%s\n", *err == CL_SUCCESS ? "log" : "FAILED", log.data());
    if (*err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return nullptr;
    }
    return prog;
}

}  // namespace

Gemma4ModelRuntime::Gemma4ModelRuntime() = default;
Gemma4ModelRuntime::~Gemma4ModelRuntime() {
    Destroy();
}

void* Gemma4ModelRuntime::Alloc(size_t bytes) {
    cl_int err = CL_SUCCESS;
    void* p = usm_.alloc(ctx_, dev_, nullptr, std::max<size_t>(bytes, 64), 0, &err);
    return err == CL_SUCCESS ? p : nullptr;
}

void* Gemma4ModelRuntime::Grow(void*& buf, size_t& cap, size_t bytes) {
    if (cap >= bytes && buf)
        return buf;
    if (buf)
        usm_.blocking_free(ctx_, buf);
    cap = round_up(bytes + bytes / 4, 4096);
    buf = Alloc(cap);
    if (!buf)
        cap = 0;
    return buf;
}

int Gemma4ModelRuntime::Init(const Gemma4ConstantParams& w, cl_context ctx, cl_device_id dev, cl_command_queue q,
                             const Usm& usm) {
    ctx_ = ctx;
    dev_ = dev;
    q_ = q;
    usm_ = usm;
    w_ = w;
    nl_ = w.layers;
    if (nl_ <= 0 || nl_ > kModelMaxLayers || !usm_.memcpy || !usm_.blocking_free) {
        std::fprintf(stderr, "[Gemma4Model] bad model description (%d layers)\n", nl_);
        return CL_INVALID_VALUE;
    }

    // Scalars live in device constants; read them back once.
    std::vector<float> meta(kMetaLayer0 + nl_);
    float mm[16];
    MK_OK(usm_.memcpy(q_, CL_TRUE, meta.data(), w.model[kModelMeta], meta.size() * 4, 0, nullptr, nullptr),
          "read model meta");
    MK_OK(usm_.memcpy(q_, CL_TRUE, mm, w.model[kModelMaskMeta], sizeof(mm), 0, nullptr, nullptr), "read mask meta");
    vocab_ = static_cast<int>(meta[kMetaVocab]);
    cap_div_ = meta[kMetaCapDiv];
    cap_mul_ = meta[kMetaCapMul];
    scale_ = meta[kMetaAttnScale];
    last_only_ = meta[kMetaLastOnly] != 0.0f;
    p_mul_ = mm[2];
    p_add_ = mm[3];
    window_ = mm[4];
    causal_off_ = mm[5];
    am_stride_ = static_cast<int>(mm[8]);

    layers_.resize(nl_);
    int hq_max = 0, hq_min = 1 << 30, d_max = 0, hm_max = 0;
    for (int l = 0; l < nl_; ++l) {
        int am[16];
        MK_OK(usm_.memcpy(q_, CL_TRUE, am, w.layer[l][kAttnMeta - kAttnWIn], sizeof(am), 0, nullptr, nullptr),
              "read attention meta");
        Layer& L = layers_[l];
        L.HQ = am[0];
        L.HK = am[1];
        L.D = am[2];
        L.RN = am[3];
        L.k_off = am[4];
        L.v_off = am[5];
        L.q_off = am[6];
        pos_stride_ = am[7];
        L.pos_f16 = am[8];
        L.half_c = am[9] / 2;
        L.R = w.layer_qkv_rows[l];
        L.sliding = meta[kMetaLayer0 + l] != 0.0f;
        const int gps = (L.D == 256 || L.D == 512) ? 512 / L.D : 0;
        const int G = L.HK ? L.HQ / L.HK : 0;
        const int qsplit = gps ? G / gps : 0;
        // mk_decode instantiates the attention for full rotary heads of 256 / 512.
        if (!gps || !G || L.HQ % L.HK || G % gps || !qsplit || 16 % qsplit || L.HK > 8 || L.RN != L.D ||
            L.R > kRMax || L.R % 16 || L.half_c <= 0) {
            std::fprintf(stderr, "[Gemma4Model] layer %d: unsupported attention HQ=%d HK=%d D=%d RN=%d R=%d\n", l,
                         L.HQ, L.HK, L.D, L.RN, L.R);
            return CL_INVALID_VALUE;
        }
        hq_max = std::max(hq_max, L.HQ);
        hq_min = std::min(hq_min, L.HQ);
        d_max = std::max(d_max, L.D);
        hm_max = std::max(hm_max, (L.HQ + 2 * L.HK) * L.D);
        int& mk = max_kv_heads_[L.sliding ? 0 : 1];
        mk = std::max(mk, L.HK);
    }

    // 256 registers per thread (the attention phase spills with 128) leave
    // room for two 256-wide work-groups per Xe-core.
    const bool big_grf = env_int("OV_MEGAKERNEL_GRF", 256) == 256;
    cl_uint cus = 0;
    clGetDeviceInfo(dev_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
    const int xe_cores = std::max(1, static_cast<int>(cus) / 8);
    nwg_ = env_int("OV_MEGAKERNEL_NWG", xe_cores * (big_grf ? 2 : 4));
    if (nwg_ < kExperts / 16 + 1 || nwg_ < max_kv_heads_[0] || nwg_ < max_kv_heads_[1]) {
        std::fprintf(stderr, "[Gemma4Model] OV_MEGAKERNEL_NWG=%d too small\n", nwg_);
        return CL_INVALID_VALUE;
    }

    char opts[512];
    std::snprintf(opts, sizeof(opts),
                  "-DH_=%d -DI_=%d -DIE_=%d -DE_=%d -DTOPK_=%d -DEPS_=1e-6f -DNL_=%d -DNWG=%d -DV_=%d %s",
                  kHidden, kInter, kExpertInter, kExperts, kTopK, nl_, nwg_, vocab_, kFfnBuildOpts);
    std::string o = opts;
    if (big_grf)
        o += " -cl-intel-256-GRF-per-thread";
    prof_on_ = env_int("OV_MEGAKERNEL_PROF", 0) != 0;
    debug_cmp_ = env_int("OV_MEGAKERNEL_DEBUG_CMP", 0) != 0;
    if (prof_on_)
        o += " -DMK_PROF";
    else if (debug_cmp_)
        o += " -DMK_DEBUG";
    if (const char* extra = std::getenv("OV_MEGAKERNEL_CL_OPTS"))
        o += std::string(" ") + extra;
    cl_int err = CL_SUCCESS;
    prog_ = build_model_program(ctx_, dev_, o, &err);
    MK_OK(err, "build model program");
    struct {
        cl_kernel* k;
        const char* name;
    } ks[] = {{&kDecode_, "mk_decode"},     {&kCvt_, "mk_cvt_embeds"}, {&kKvAppend_, "mk_kv_append"},
              {&kMaskRange_, "mk_mask_range"}, {&kPfAttn_, "mk_pf_attn"}, {&kPfFa_, "mk_pf_fa"},  {&kLmRows_, "mk_lm_head_rows"}};
    for (auto& k : ks) {
        *k.k = clCreateKernel(prog_, k.name, &err);
        MK_OK(err, k.name);
        if (std::getenv("OV_MEGAKERNEL_DUMP")) {
            cl_ulong spill = 0, slm = 0, priv = 0;
            clGetKernelWorkGroupInfo(*k.k, dev_, 0x4109 /*CL_KERNEL_SPILL_MEM_SIZE_INTEL*/, sizeof(spill), &spill,
                                     nullptr);
            clGetKernelWorkGroupInfo(*k.k, dev_, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(slm), &slm, nullptr);
            clGetKernelWorkGroupInfo(*k.k, dev_, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(priv), &priv, nullptr);
            std::fprintf(stderr, "[Gemma4Model] kernel %s spill=%llu slm=%llu private=%llu\n", k.name,
                         (unsigned long long)spill, (unsigned long long)slm, (unsigned long long)priv);
        }
    }
    // mk_decode reaches every weight through the tables.
    cl_bool on = CL_TRUE;
    clSetKernelExecInfo(kDecode_, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL, sizeof(on), &on);
    clSetKernelExecInfo(kDecode_, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL, sizeof(on), &on);
    clSetKernelExecInfo(kDecode_, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL, sizeof(on), &on);
    clSetKernelExecInfo(kLmRows_, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL, sizeof(on), &on);

    // Decode scratch.
    const int p_max = nwg_ * 16 * 2 / std::max(1, hq_min);
    const size_t sizes[kSCount] = {
        size_t(kHidden) * 2,
        size_t(kRMax) * 8,  // fixed-point accumulators are 64-bit
        size_t(hq_max) * d_max * 2,
        size_t(kHidden) * 8,
        size_t(kHidden) * 4,
        size_t(2) * kInter * 8,
        size_t(kExperts) * 8,
        size_t(kTopK) * 2 * kExpertInter * 8,
        size_t(kHidden) * 8,
        size_t(kHidden) * 8,
        size_t(hq_max) * p_max * (d_max + 16) * 4,
        size_t(nl_ * kPhasesPerLayer + 1 + nl_) * 4,
        size_t(nl_) * 8 * 4,
        4,
        size_t(kTopK * kExpertInter / 2 + 2 * kTopK * kExpertInter / 64) * 4,
        size_t(kInter / 2 + 2 * kInter / 64) * 4,
    };
    size_t total = 0;
    for (int i = 0; i < kSCount; ++i) {
        off_[i] = total;
        total += round_up(sizes[i], 256);
    }
    scratch_ = Alloc(total);
    if (!scratch_)
        return CL_OUT_OF_RESOURCES;
    {
        std::vector<uint8_t> zeros(total, 0);
        MK_OK(usm_.memcpy(q_, CL_TRUE, scratch_, zeros.data(), total, 0, nullptr, nullptr), "zero scratch");
    }
    tab_ = Alloc(sizeof(ModelTabH));
    ltab_ = Alloc(sizeof(LayerTabH) * nl_);
    if (!tab_ || !ltab_)
        return CL_OUT_OF_RESOURCES;
    if (debug_cmp_ && !prof_on_) {
        prof_ = Alloc(size_t(nl_ + 1) * kHidden * 2 + size_t(nl_) * 8192 * 2);
        if (!prof_)
            return CL_OUT_OF_RESOURCES;
    }
    if (prof_on_) {
        prof_ = Alloc((size_t(nl_ * kPhasesPerLayer + 2) * nwg_ * 5 + size_t(nl_) * 16) * 8);
        if (!prof_)
            return CL_OUT_OF_RESOURCES;
    }

    const int rc = EnsureCapacity(env_int("OV_MEGAKERNEL_KV_CAP", 4096));
    if (rc != CL_SUCCESS)
        return rc;

    MK_OK(usm_.set_arg(kDecode_, 0, tab_), "mk_decode tab");
    MK_OK(usm_.set_arg(kDecode_, 1, ltab_), "mk_decode layers");

    // Prompt path: per-layer runtimes over the same device weights.
    Gemma4PlatformParams pp{};
    pp.deviceId = dev_;
    pp.context = ctx_;
    pp.stream = q_;
    for (int l = 0; l < nl_; ++l) {
        Gemma4ConstantParams a{};
        a.kind = kAttnIn;
        for (int p = 0; p < kModelAttnPorts; ++p)
            a.attn[kAttnWIn + p] = w.layer[l][p];
        a.attn_rows = layers_[l].R;
        attn_rt_.emplace_back(new Gemma4POCRuntime());
        MK_OK(attn_rt_.back()->Init(&a, &pp), "attention runtime");
        Gemma4ConstantParams f{};
        f.kind = kFfn;
        for (int p = 0; p < kModelFfnPorts; ++p)
            f.ffn[kFfnWPostAttn + p] = w.layer[l][kModelAttnPorts + p];
        ffn_rt_.emplace_back(new Gemma4POCRuntime());
        MK_OK(ffn_rt_.back()->Init(&f, &pp), "ffn runtime");
    }
    for (int kind : {kMaskFull, kMaskSliding}) {
        Gemma4ConstantParams m{};
        m.kind = kind;
        m.mask_meta = w.model[kModelMaskMeta];
        mask_rt_.emplace_back(new Gemma4POCRuntime());
        MK_OK(mask_rt_.back()->Init(&m, &pp), "mask runtime");
    }
    (void)hm_max;
    if (std::getenv("OV_MEGAKERNEL_DUMP"))
        std::fprintf(stderr,
                     "[Gemma4Model] %d layers, V=%d, NWG=%d, softcap %g/%g, scale %g, window %g, last_only %d, "
                     "pos stride %d, am stride %d\n",
                     nl_, vocab_, nwg_, cap_div_, cap_mul_, scale_, window_, last_only_, pos_stride_, am_stride_);
    return CL_SUCCESS;
}

int Gemma4ModelRuntime::UploadTables() {
    hltab_.assign(sizeof(LayerTabH) * nl_, 0);
    auto* lt = reinterpret_cast<LayerTabH*>(hltab_.data());
    for (int l = 0; l < nl_; ++l) {
        void* const* a = w_.layer[l];
        void* const* f = w_.layer[l] + kModelAttnPorts - kFfnWPostAttn;  // indexed by FfnPort
        const Layer& L = layers_[l];
        LayerTabH& t = lt[l];
        t.w_in = a[kAttnWIn - kAttnWIn];
        t.qkv = a[kAttnQkv - kAttnWIn];
        t.gq = a[kAttnWQ - kAttnWIn];
        t.gk = a[kAttnWK - kAttnWIn];
        t.invf = a[kAttnInvFreq - kAttnWIn];
        t.oproj = f[kFfnOProj];
        t.w_pa = f[kFfnWPostAttn];
        t.w_pf = f[kFfnWPreFf];
        t.w_r = f[kFfnWRouter];
        t.w_pf2 = f[kFfnWPreFf2];
        t.gate = f[kFfnGate];
        t.up = f[kFfnUp];
        t.down = f[kFfnDown];
        t.rw = f[kFfnRouterW];
        t.egate = f[kFfnEGate];
        t.eup = f[kFfnEUp];
        t.edown = f[kFfnEDown];
        t.w_pf1 = f[kFfnWPostFf1];
        t.w_pf2p = f[kFfnWPostFf2];
        t.w_post = f[kFfnWPostFf];
        t.ls = f[kFfnLayerScalar];
        t.kc = L.kc;
        t.vc = L.vc;
        t.R = L.R;
        t.HQ = L.HQ;
        t.HK = L.HK;
        t.D = L.D;
        t.RN = L.RN;
        t.q_off = L.q_off;
        t.k_off = L.k_off;
        t.v_off = L.v_off;
        t.half_c = L.half_c;
        t.pos_f16 = L.pos_f16;
        t.sliding = L.sliding;
    }
    htab_.assign(sizeof(ModelTabH), 0);
    auto* mt = reinterpret_cast<ModelTabH*>(htab_.data());
    auto* s = static_cast<uint8_t*>(scratch_);
    void** ptrs[kSCount] = {&mt->xbuf, &mt->raw,  &mt->att,  &mt->o,   &mt->h1,  &mt->gu,  &mt->lg,  &mt->egu,
                            &mt->m,    &mt->e,    &mt->part, &mt->sync, &mt->tick, &mt->err, &mt->xqm, &mt->xqd};
    for (int i = 0; i < kSCount; ++i)
        *ptrs[i] = s + off_[i];
    mt->w_final = w_.model[kModelFinalNorm];
    mt->lm = w_.model[kModelLmW];
    mt->lm_s = w_.model[kModelLmScale];
    mt->lm_z = w_.model[kModelLmZp];
    mt->cap_div = cap_div_;
    mt->cap_mul = cap_mul_;
    mt->attn_scale = scale_;
    mt->window = window_;
    mt->causal_off = causal_off_;
    mt->p_mul = p_mul_;
    mt->p_add = p_add_;
    mt->pos_stride = pos_stride_;
    mt->am_stride = am_stride_;
    mt->kv_cap = kv_cap_;
    mt->prof = prof_;
    MK_OK(usm_.memcpy(q_, CL_TRUE, ltab_, hltab_.data(), hltab_.size(), 0, nullptr, nullptr), "upload layer table");
    MK_OK(usm_.memcpy(q_, CL_TRUE, tab_, htab_.data(), htab_.size(), 0, nullptr, nullptr), "upload model table");
    return CL_SUCCESS;
}

int Gemma4ModelRuntime::EnsureCapacity(int L) {
    if (L <= kv_cap_)
        return CL_SUCCESS;
    const int cap = static_cast<int>(round_up(std::max(L, kv_cap_ * 2), 1024));
    for (auto& ly : layers_) {
        const size_t row = size_t(ly.D) * 2;
        for (void** buf : {&ly.kc, &ly.vc}) {
            void* nb = Alloc(size_t(ly.HK) * cap * row);
            if (!nb)
                return CL_OUT_OF_RESOURCES;
            if (*buf) {
                for (int h = 0; h < ly.HK && kv_valid_ > 0; ++h)
                    MK_OK(usm_.memcpy(q_, CL_FALSE, static_cast<uint8_t*>(nb) + size_t(h) * cap * row,
                                      static_cast<uint8_t*>(*buf) + size_t(h) * kv_cap_ * row, kv_valid_ * row, 0,
                                      nullptr, nullptr),
                          "kv copy");
                usm_.blocking_free(ctx_, *buf);
            }
            *buf = nb;
        }
    }
    kv_cap_ = cap;
    return UploadTables();
}

int Gemma4ModelRuntime::Execute(const Gemma4RuntimeParams& p) {
    const int T = p.tokens, L = p.ka;
    if (p.batch != 1 || T <= 0 || L < T) {
        std::fprintf(stderr, "[Gemma4Model] unsupported call: batch %d, tokens %d, attention_mask %d\n", p.batch, T,
                     L);
        return CL_INVALID_VALUE;
    }
    const int past = L - T;
    if (past > kv_valid_) {
        std::fprintf(stderr, "[Gemma4Model] past length %d exceeds the cached %d tokens\n", past, kv_valid_);
        return CL_INVALID_VALUE;
    }
    const int rc = EnsureCapacity(L);
    if (rc != CL_SUCCESS)
        return rc;
    // OV_MEGAKERNEL_NO_DECODE=1 runs single tokens through the prompt path
    // (debugging aid: the per-layer kernels instead of mk_decode).
    static const bool no_decode = env_int("OV_MEGAKERNEL_NO_DECODE", 0) != 0;
    if (T == 1 && debug_cmp_ && !prof_on_) {
        // Residual entering every layer, prompt path vs mk_decode.
        std::vector<uint16_t> ref(size_t(nl_) * kHidden + size_t(nl_) * 8192),
            got(size_t(nl_) * kHidden + size_t(nl_) * 8192);
        dbg_x_ = &ref;
        int rc0 = Prompt(p, past, L);
        dbg_x_ = nullptr;
        if (rc0 == CL_SUCCESS)
            rc0 = Decode(p, past, L);
        clFinish(q_);
        usm_.memcpy(q_, CL_TRUE, got.data(), prof_, got.size() * 2, 0, nullptr, nullptr);
        auto h2f = [](uint16_t h) {
            const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
            float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? INFINITY : std::ldexp(float(m | 1024), int(e) - 25);
            return s ? -v : v;
        };
        std::fprintf(stderr, "[Gemma4Model] past %d: per-layer residual cos (prompt path vs mk_decode):", past);
        for (int l = 0; l < nl_; ++l) {
            double ab = 0, aa = 0, bb = 0;
            for (int i = 0; i < kHidden; ++i) {
                const double a = h2f(ref[size_t(l) * kHidden + i]), b = h2f(got[size_t(l) * kHidden + i]);
                ab += a * b;
                aa += a * a;
                bb += b * b;
            }
            std::fprintf(stderr, " %d:%.4f", l, ab / std::sqrt(aa * bb + 1e-30));
        }
        std::fprintf(stderr, "\n  attention output cos:");
        for (int l = 0; l < nl_; ++l) {
            const int ka = layers_[l].HQ * layers_[l].D;
            double ab = 0, aa = 0, bb = 0;
            for (int i = 0; i < ka; ++i) {
                const size_t o = size_t(nl_) * kHidden + size_t(l) * 8192 + i;
                const double a = h2f(ref[o]), b = h2f(got[o]);
                ab += a * b;
                aa += a * a;
                bb += b * b;
            }
            std::fprintf(stderr, " %d:%.4f", l, ab / std::sqrt(aa * bb + 1e-30));
        }
        for (int l = 0; l < nl_; ++l) {
            if (layers_[l].sliding)
                continue;
            const int D = layers_[l].D;
            std::fprintf(stderr, "\n  layer %d per-head cos:", l);
            for (int h = 0; h < layers_[l].HQ; ++h) {
                double ab = 0, aa = 0, bb = 0;
                for (int i = 0; i < D; ++i) {
                    const size_t o = size_t(nl_) * kHidden + size_t(l) * 8192 + size_t(h) * D + i;
                    const double a = h2f(ref[o]), b = h2f(got[o]);
                    ab += a * b;
                    aa += a * a;
                    bb += b * b;
                }
                std::fprintf(stderr, " %d:%.4f", h, ab / std::sqrt(aa * bb + 1e-30));
            }
            break;
        }
        std::fprintf(stderr, "\n");
        if (rc0 == CL_SUCCESS)
            kv_valid_ = L;
        return rc0;
    }
    const int r = T == 1 && !no_decode ? Decode(p, past, L) : Prompt(p, past, L);
    if (r == CL_SUCCESS)
        kv_valid_ = L;
    return r;
}

int Gemma4ModelRuntime::Decode(const Gemma4RuntimeParams& p, int past, int L) {
    // Key splits per kv head: a work-group has 16 (sliding) or 2 (full) key
    // lanes; aim for OV_MEGAKERNEL_ATT_KPS keys per lane.
    static const int kps_s = env_int("OV_MEGAKERNEL_ATT_KPS_S", 2);
    static const int kps_f = env_int("OV_MEGAKERNEL_ATT_KPS_F", 4);
    auto splits = [&](int nk, int keys_per_unit, int hk) {
        const int lim = hk ? nwg_ / hk : 1;
        return std::max(1, std::min(lim, (nk + keys_per_unit - 1) / keys_per_unit));
    };
    const int ns_s = splits(std::min(static_cast<int>(window_), past + 1), 16 * kps_s, max_kv_heads_[0]);
    const int ns_f = splits(past + 1, 2 * kps_f, max_kv_heads_[1]);
    ++epoch_;
    cl_kernel k = kDecode_;
    MK_OK(usm_.set_arg(k, 2, p.in), "embeds");
    MK_OK(usm_.set_arg(k, 3, p.in2), "pos");
    MK_OK(usm_.set_arg(k, 4, p.in3), "am");
    MK_OK(usm_.set_arg(k, 5, p.out), "logits");
    MK_OK(clSetKernelArg(k, 6, sizeof(int), &past), "past");
    MK_OK(clSetKernelArg(k, 7, sizeof(int), &L), "L");
    MK_OK(clSetKernelArg(k, 8, sizeof(int), &epoch_), "epoch");
    MK_OK(clSetKernelArg(k, 9, sizeof(int), &ns_s), "ns_s");
    MK_OK(clSetKernelArg(k, 10, sizeof(int), &ns_f), "ns_f");
    // Attention unit counters grow by NS per launch (all layers of a type alike).
    MK_OK(clSetKernelArg(k, 11, sizeof(int), &att_base_[0]), "att_base_s");
    MK_OK(clSetKernelArg(k, 12, sizeof(int), &att_base_[1]), "att_base_f");
    att_base_[0] += ns_s > 1 ? ns_s : 0;
    att_base_[1] += ns_f > 1 ? ns_f : 0;
    const size_t gws = size_t(nwg_) * kLws, lws = kLws;
    MK_OK(clEnqueueNDRangeKernel(q_, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr), "mk_decode");
    if (prof_on_)
        Profile();
    return CL_SUCCESS;
}

// OV_MEGAKERNEL_PROF=1: per-phase critical path from the barrier timestamps.
void Gemma4ModelRuntime::Profile() {
    const int nb = nl_ * kPhasesPerLayer + 2;
    std::vector<uint64_t> t(size_t(nb) * nwg_ * 5 + size_t(nl_) * 16);
    clFinish(q_);
    usm_.memcpy(q_, CL_TRUE, t.data(), prof_, t.size() * 8, 0, nullptr, nullptr);
    auto at = [&](int b, int w, int which) { return double(t[(size_t(b) * nwg_ + w) * 5 + which]); };
    const int ks = nb - 1;  // kernel start / end slot
    if (prof_acc_.empty())
        prof_acc_.assign(5 * (kPhasesPerLayer + 3), 0.0);
    // Phase p of a layer ends at barrier p; G ends at the kernel end.
    for (int b = 0; b <= nl_ * kPhasesPerLayer; ++b) {
        const bool last = b == nl_ * kPhasesPerLayer;
        int ph = last ? kPhasesPerLayer : b % kPhasesPerLayer;
        if (ph == 1 && !layers_[b / kPhasesPerLayer].sliding)
            ph = kPhasesPerLayer + 2;  // attention of the full-attention layers
        double start_min = 1e30, arrive_max = 0, work = 0, release_min = 1e30, release_max = 0, pro = 0;
        for (int w = 0; w < nwg_; ++w) {
            const double s = b == 0 ? at(ks, w, 0) : at(b - 1, w, 1);
            const double a = last ? at(ks, w, 1) : at(b, w, 0);
            start_min = std::min(start_min, s);
            arrive_max = std::max(arrive_max, a);
            work += a - s;
            const double ws = at(last ? ks - 1 : b, w, 2);
            pro += ws > s && ws < a ? ws - s : 0;
            if (!last) {
                release_min = std::min(release_min, at(b, w, 1));
                release_max = std::max(release_max, at(b, w, 1));
            }
        }
        double* acc = &prof_acc_[5 * ph];
        acc[0] += arrive_max - start_min;                       // phase span (first start .. last arrival)
        acc[1] += work / nwg_;                                  // mean work-group busy time
        acc[2] += last ? 0 : release_min - arrive_max;          // barrier latency
        acc[3] += last ? 0 : release_max - release_min;         // release skew
        acc[4] += pro / nwg_;                                   // mean prologue (before streaming)
    }
    double tot = 0;
    for (int w = 0; w < nwg_; ++w)
        tot = std::max(tot, at(ks, w, 1) - at(ks, 0, 0));
    prof_acc_[5 * (kPhasesPerLayer + 1)] += tot;
    // Phase D: dense part, router wait, MoE part (mean over work-groups).
    if (prof_d_.empty())
        prof_d_.assign(4, 0.0);
    for (int l = 0; l < nl_; ++l) {
        const int b = l * kPhasesPerLayer + 3;
        for (int w = 0; w < nwg_; ++w) {
            prof_d_[0] += at(b, w, 3) - at(b, w, 2);
            prof_d_[1] += at(b, w, 4) - at(b, w, 3);
            prof_d_[2] += at(b, w, 0) - at(b, w, 4);
        }
    }
    if (prof_dbg_.empty())
        prof_dbg_.assign(16, 0.0);
    for (int l = 0; l < nl_; ++l) {
        const uint64_t* d = &t[size_t(nb) * nwg_ * 5 + size_t(l) * 16];
        const int pts[5] = {0, 3, 4, 5, 6};
        for (int i = 1; i < 5; ++i)
            prof_dbg_[(layers_[l].sliding ? 0 : 8) + i] += double(d[pts[i]] - d[pts[i - 1]]);
    }
    if (++prof_n_ % 20)
        return;
    const char* names[] = {"A qkv", "B attn(s)", "C oproj", "D gate/up", "E0 act", "E down", "G lm_head", "",
                           "B attn(f)"};
    int nfull = 0;
    for (const auto& ly : layers_)
        nfull += !ly.sliding;
    const double n = prof_n_;
    std::fprintf(stderr, "[Gemma4Model] profile over %d steps (timestamp ticks, per layer for A..F):\n", prof_n_);
    for (int ph = 0; ph <= kPhasesPerLayer + 2; ++ph) {
        if (ph == kPhasesPerLayer + 1)
            continue;
        const double div = n * (ph == kPhasesPerLayer ? 1 : ph == 1 ? nl_ - nfull : ph > kPhasesPerLayer ? nfull : nl_);
        const double* a = &prof_acc_[5 * ph];
        std::fprintf(stderr, "  %-10s span %7.2f us  mean-busy %7.2f  prologue %6.2f  barrier %5.2f  skew %5.2f\n",
                     names[ph], a[0] / div / 2400, a[1] / div / 2400, a[4] / div / 2400, a[2] / div / 2400,
                     a[3] / div / 2400);
    }
    std::fprintf(stderr, "  unit 0 (sliding | full) us: table+q, keys, wg combine, (last unit) merge:");
    for (int i = 1; i < 5; ++i)
        std::fprintf(stderr, " %.2f|%.2f", prof_dbg_[i] / n / (nl_ - nfull) / 2400, prof_dbg_[8 + i] / n / nfull / 2400);
    std::fprintf(stderr, "\n");
    prof_dbg_.assign(16, 0.0);
    std::fprintf(stderr, "  D: dense/router %.2f us, routing wait %.2f, MoE gate/up %.2f (work-group mean)\n",
                 prof_d_[0] / n / nl_ / nwg_ / 2400, prof_d_[1] / n / nl_ / nwg_ / 2400,
                 prof_d_[2] / n / nl_ / nwg_ / 2400);
    prof_d_.assign(4, 0.0);
    std::fprintf(stderr, "  kernel total %.3f ms (2.4 GHz ticks)\n", prof_acc_[5 * (kPhasesPerLayer + 1)] / n / 2.4e6);
    prof_acc_.assign(prof_acc_.size(), 0.0);
    prof_n_ = 0;
}

int Gemma4ModelRuntime::Prompt(const Gemma4RuntimeParams& p, int past, int L) {
    const int T = p.tokens;
    int hm_max = 0, o_max = 0;
    for (const auto& ly : layers_) {
        hm_max = std::max(hm_max, (ly.HQ + 2 * ly.HK) * ly.D);
        o_max = std::max(o_max, ly.HQ * ly.D);
    }
    const size_t need[kPfBufs] = {size_t(T) * kHidden * 2, size_t(T) * kHidden * 2, size_t(T) * hm_max * 2,
                                  size_t(T) * o_max * 2,   size_t(T) * L * 2,       size_t(T) * L * 2,
                                  size_t(T) * 4 * 4};
    for (int i = 0; i < kPfBufs; ++i)
        if (!Grow(pf_[i], pf_cap_[i], need[i]))
            return CL_OUT_OF_RESOURCES;
    void* xa = pf_[kXa];
    void* xb = pf_[kXb];
    int* rng = static_cast<int*>(pf_[kRange]);
    int* lo[2] = {rng, rng + 2 * T};
    int* hi[2] = {rng + T, rng + 3 * T};

    {
        const int n = T * kHidden;
        MK_OK(usm_.set_arg(kCvt_, 0, p.in), "cvt in");
        MK_OK(usm_.set_arg(kCvt_, 1, xa), "cvt out");
        MK_OK(clSetKernelArg(kCvt_, 2, sizeof(int), &n), "cvt n");
        const size_t gws = round_up(n, 256), lws = 256;
        MK_OK(clEnqueueNDRangeKernel(q_, kCvt_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr), "mk_cvt_embeds");
    }
    // Explicit [T, L] masks (vision groups included), and their visible key ranges.
    for (int i = 0; i < 2; ++i) {
        Gemma4RuntimeParams mp{};
        mp.in = p.in3;
        mp.in2 = p.in4;
        mp.out = pf_[i == 0 ? kMaskF : kMaskS];
        mp.tokens = 1;
        mp.ka = L;
        mp.s = T;
        const int rc = mask_rt_[i]->Execute(&mp);
        if (rc != CL_SUCCESS)
            return rc;
        MK_OK(usm_.set_arg(kMaskRange_, 0, mp.out), "range mask");
        MK_OK(clSetKernelArg(kMaskRange_, 1, sizeof(int), &T), "range T");
        MK_OK(clSetKernelArg(kMaskRange_, 2, sizeof(int), &L), "range L");
        MK_OK(usm_.set_arg(kMaskRange_, 3, lo[i]), "range lo");
        MK_OK(usm_.set_arg(kMaskRange_, 4, hi[i]), "range hi");
        const size_t gws[2] = {16, size_t(T)}, lws[2] = {16, 1};
        MK_OK(clEnqueueNDRangeKernel(q_, kMaskRange_, 2, nullptr, gws, lws, 0, nullptr, nullptr), "mk_mask_range");
    }

    for (int l = 0; l < nl_; ++l) {
        const Layer& ly = layers_[l];
        if (dbg_x_)
            usm_.memcpy(q_, CL_TRUE, dbg_x_->data() + size_t(l) * kHidden,
                        static_cast<uint8_t*>(xa) + size_t(T - 1) * kHidden * 2, kHidden * 2, 0, nullptr, nullptr);
        Gemma4RuntimeParams ap{};
        ap.in = xa;
        ap.in2 = p.in2;
        ap.out = pf_[kHm];
        ap.tokens = T;
        ap.s = T;
        int rc = attn_rt_[l]->Execute(&ap);
        if (rc != CL_SUCCESS)
            return rc;
        {
            cl_kernel k = kKvAppend_;
            MK_OK(usm_.set_arg(k, 0, pf_[kHm]), "kv hm");
            MK_OK(clSetKernelArg(k, 1, sizeof(int), &T), "kv T");
            MK_OK(clSetKernelArg(k, 2, sizeof(int), &ly.HQ), "kv HQ");
            MK_OK(clSetKernelArg(k, 3, sizeof(int), &ly.HK), "kv HK");
            MK_OK(clSetKernelArg(k, 4, sizeof(int), &ly.D), "kv D");
            MK_OK(usm_.set_arg(k, 5, ly.kc), "kv kc");
            MK_OK(usm_.set_arg(k, 6, ly.vc), "kv vc");
            MK_OK(clSetKernelArg(k, 7, sizeof(int), &kv_cap_), "kv cap");
            MK_OK(clSetKernelArg(k, 8, sizeof(int), &past), "kv past");
            const size_t gws[3] = {round_up(ly.D, 64), size_t(T), size_t(2 * ly.HK)}, lws[3] = {64, 1, 1};
            MK_OK(clEnqueueNDRangeKernel(q_, k, 3, nullptr, gws, lws, 0, nullptr, nullptr), "mk_kv_append");
        }
        {
            // XMX flash attention; OV_MEGAKERNEL_PF_ATTN_SIMPLE=1 keeps the scalar kernel.
            static const bool simple = std::getenv("OV_MEGAKERNEL_PF_ATTN_SIMPLE") != nullptr;
            cl_kernel k = simple ? kPfAttn_ : kPfFa_;
            const int s = ly.sliding ? 1 : 0;
            MK_OK(usm_.set_arg(k, 0, pf_[kHm]), "attn hm");
            MK_OK(usm_.set_arg(k, 1, ly.kc), "attn kc");
            MK_OK(usm_.set_arg(k, 2, ly.vc), "attn vc");
            MK_OK(usm_.set_arg(k, 3, pf_[s ? kMaskS : kMaskF]), "attn mask");
            MK_OK(usm_.set_arg(k, 4, lo[s]), "attn lo");
            MK_OK(usm_.set_arg(k, 5, hi[s]), "attn hi");
            MK_OK(clSetKernelArg(k, 6, sizeof(int), &T), "attn T");
            MK_OK(clSetKernelArg(k, 7, sizeof(int), &L), "attn L");
            MK_OK(clSetKernelArg(k, 8, sizeof(int), &ly.HQ), "attn HQ");
            MK_OK(clSetKernelArg(k, 9, sizeof(int), &ly.HK), "attn HK");
            MK_OK(clSetKernelArg(k, 10, sizeof(int), &ly.D), "attn D");
            MK_OK(clSetKernelArg(k, 11, sizeof(int), &kv_cap_), "attn cap");
            MK_OK(clSetKernelArg(k, 12, sizeof(float), &scale_), "attn scale");
            MK_OK(usm_.set_arg(k, 13, pf_[kO]), "attn O");
            const int qb = simple ? 512 / ly.D : 8;
            const int per_wg = simple ? 16 : 8;
            const size_t sgs = size_t(ly.HQ) * ((T + qb - 1) / qb) * (simple ? 1 : ly.D / 256);
            const size_t gws = round_up(sgs, per_wg) * 16, lws = per_wg * 16;
            MK_OK(clEnqueueNDRangeKernel(q_, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr), "mk_pf_attn");
        }
        if (dbg_x_)
            usm_.memcpy(q_, CL_TRUE, dbg_x_->data() + size_t(nl_) * kHidden + size_t(l) * 8192,
                        static_cast<uint8_t*>(pf_[kO]) + size_t(T - 1) * ly.HQ * ly.D * 2, size_t(ly.HQ) * ly.D * 2,
                        0, nullptr, nullptr);
        Gemma4RuntimeParams fp{};
        fp.in = pf_[kO];
        fp.in2 = xa;
        fp.out = xb;
        fp.tokens = T;
        fp.ka = ly.HQ * ly.D;
        rc = ffn_rt_[l]->Execute(&fp);
        if (rc != CL_SUCCESS)
            return rc;
        std::swap(xa, xb);
    }

    const int rows = last_only_ ? 1 : T;
    {
        cl_kernel k = kLmRows_;
        void* x = static_cast<uint8_t*>(xa) + size_t(T - rows) * kHidden * 2;
        MK_OK(usm_.set_arg(k, 0, tab_), "lm tab");
        MK_OK(usm_.set_arg(k, 1, x), "lm x");
        MK_OK(clSetKernelArg(k, 2, sizeof(int), &rows), "lm rows");
        MK_OK(usm_.set_arg(k, 3, p.out), "lm out");
        const size_t wgs = (size_t(vocab_) / 16 + 16 * kLmRb - 1) / (16 * kLmRb);
        const size_t gws[2] = {size_t((rows + kLmTb - 1) / kLmTb) * kLws, wgs}, lws[2] = {kLws, 1};
        MK_OK(clEnqueueNDRangeKernel(q_, k, 2, nullptr, gws, lws, 0, nullptr, nullptr), "mk_lm_head_rows");
    }
    return CL_SUCCESS;
}

void Gemma4ModelRuntime::Destroy() {
    attn_rt_.clear();
    ffn_rt_.clear();
    mask_rt_.clear();
    for (cl_kernel* k : {&kDecode_, &kCvt_, &kKvAppend_, &kMaskRange_, &kPfAttn_, &kPfFa_, &kLmRows_}) {
        if (*k)
            clReleaseKernel(*k);
        *k = nullptr;
    }
    if (prog_)
        clReleaseProgram(prog_);
    prog_ = nullptr;
    if (!ctx_ || !usm_.blocking_free)
        return;
    // In-flight kernels may still use these; the blocking free waits for them.
    for (auto& ly : layers_)
        for (void** b : {&ly.kc, &ly.vc})
            if (*b) {
                usm_.blocking_free(ctx_, *b);
                *b = nullptr;
            }
    for (void** b : {&scratch_, &tab_, &ltab_, &prof_})
        if (*b) {
            usm_.blocking_free(ctx_, *b);
            *b = nullptr;
        }
    for (auto& b : pf_)
        if (b) {
            usm_.blocking_free(ctx_, b);
            b = nullptr;
        }
}

}  // namespace mk
