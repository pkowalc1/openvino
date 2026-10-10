#pragma once

#include <CL/cl_ext.h>

#include "../qwen06BPOCParams.h"
#include "taskSystem/host/taskManagerHost.h"

namespace mk {

// TEMPORARY HERE:
struct MonoCtxH {
    void* hs = nullptr;
    void* h = nullptr;
    void* out = nullptr;
    void* wn = nullptr;
    void* pn = nullptr;
    void* qw = nullptr;
    void* kw = nullptr;
    void* vw = nullptr;
    void* ow = nullptr;
    void* gw = nullptr;
    void* uw = nullptr;
    void* dw = nullptr;
    void* qn = nullptr;
    void* kn = nullptr;
    void* rf = nullptr;
    void* qb = nullptr;
    void* kb = nullptr;
    void* vb = nullptr;
    void* xn = nullptr;
    void* gbuf = nullptr;
    void* nbuf = nullptr;
    void* kc = nullptr;
    void* vc = nullptr;
    void* sync = nullptr;
    void* past_pos = nullptr;
    void* attn_partials = nullptr;
    int step = 0;
    unsigned CS = 0;
    unsigned tok_off = 0;
};

class Qwen06BPOCRuntime : public IMegakernelRuntime {
public:
    Qwen06BPOCRuntime() = default;
    ~Qwen06BPOCRuntime() = default;

    TErrorcode Init(const IConstantParams* constantParams, const IPlatformParams* platformParams) override;
    TErrorcode Execute(const IRuntimeParams* runtimeParams) override;
    TErrorcode Destroy() override;

private:
    cl_context ctx_ = nullptr;
    cl_device_id dev_ = nullptr;
    cl_program prog_ = nullptr;
    cl_command_queue stream_ = nullptr;
    cl_kernel kTask_ = nullptr;  // task-system worker kernel (whole model)
    // Intel USM extension entry points (resolved in Init).
    clDeviceMemAllocINTEL_fn usmAlloc_ = nullptr;
    clMemFreeINTEL_fn usmFree_ = nullptr;
    clEnqueueMemcpyINTEL_fn usmMemcpy_ = nullptr;
    clEnqueueMemFillINTEL_fn usmMemFill_ = nullptr;
    // USM device allocations: per-token scratch, KV cache, sync counters, context.
    void* mQb_ = nullptr;
    void* mKb_ = nullptr;
    void* mVb_ = nullptr;
    void* mGb_ = nullptr;
    void* mXn_ = nullptr;
    void* mH_ = nullptr;
    void* mNb_ = nullptr;
    void* mKC_ = nullptr;
    void* mVC_ = nullptr;
    void* mSync_ = nullptr;
    void* mAttnPartials_ = nullptr;
    void* mCtx_ = nullptr;
    TaskManager taskManager_{};
    cl_mem mTaskMgr_ = nullptr;
    MonoCtxH runtimeContext_{};
};

}  // namespace mk