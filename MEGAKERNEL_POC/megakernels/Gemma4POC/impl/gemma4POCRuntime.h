#pragma once

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "../gemma4POCParams.h"

namespace mk {

// Runtime behind every Gemma-4 MegaKernel op; MegaKernelAttrs::kind selects
// what it replaces (see gemma4POCParams.h).
//
// kFfn: one decoder layer's feed-forward block, as 4 launches per token
// (gemma4FfnKernels.h). The OpenCL program is built once per context and shared
// by all 30 layer instances.
//
// kLmHead (opt-in, measured slower than stock): u8 lm_head GEMV.
class Gemma4POCRuntime : public IMegakernelRuntime {
public:
    Gemma4POCRuntime() = default;
    ~Gemma4POCRuntime() override { Destroy(); }

    TErrorcode Init(const IConstantParams* weights, const IPlatformParams* platform) override;
    TErrorcode Execute(const IRuntimeParams* io) override;
    TErrorcode Destroy() override;

private:
    TErrorcode InitFfn();
    TErrorcode ExecuteFfn(const Gemma4RuntimeParams& p);
    TErrorcode InitAttn();
    TErrorcode ExecuteAttn(const Gemma4RuntimeParams& p);
    TErrorcode InitMask();
    TErrorcode ExecuteMask(const Gemma4RuntimeParams& p);
    TErrorcode InitLmHead();
    TErrorcode ExecuteLmHead(const Gemma4RuntimeParams& p);

    cl_context ctx_ = nullptr;
    cl_device_id dev_ = nullptr;
    cl_command_queue stream_ = nullptr;
    cl_program prog_ = nullptr;  // owned only for kLmHead; kFfn uses the shared one
    cl_kernel kGemv_ = nullptr;
    cl_kernel kSumx_ = nullptr;
    // ffn_oproj, ffn_in, ffn_mid, ffn_moe_down, ffn_out (kFfnKernels).
    cl_kernel kFfn_[5] = {};
    size_t ffnGws_[5] = {};
    // attn_qkv, attn_post.
    cl_kernel kAttn_[2] = {};
    size_t attnGws_[2] = {};
    // mask_groups, mask_fill.
    cl_kernel kMask_[2] = {};
    size_t maskGrpCap_ = 0;  // ints in scratch_ for kMaskSliding

    // Resolved against the device's platform; a null platform yields null here.
    clDeviceMemAllocINTEL_fn usmAlloc_ = nullptr;
    clMemFreeINTEL_fn usmFree_ = nullptr;
    clSetKernelArgMemPointerINTEL_fn setUsmArg_ = nullptr;
    clMemBlockingFreeINTEL_fn usmBlockingFree_ = nullptr;

    Gemma4ConstantParams w_{};
    void* scratch_ = nullptr;  // kFfn: f32/i32 intermediates of one token
    void* mSumx_ = nullptr;    // kLmHead: per-token activation sums
    int scratch_tokens_ = 0;

    void ensure_scratch(int tokens);
};

}  // namespace mk
