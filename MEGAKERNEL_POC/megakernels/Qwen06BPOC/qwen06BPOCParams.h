#pragma once
#include <CL/cl.h>

#include "../iMegakernelRuntime.h"

namespace mk {
class Qwen06BConstantParams : public IConstantParams {
public:
    void* q_proj_w;
    void* k_proj_w;
    void* v_proj_w;
    void* o_proj_w;
    void* gate_proj_w;
    void* up_proj_w;
    void* down_proj_w;
    void* input_ln_w;
    void* post_attn_ln_w;
    void* q_norm_w;
    void* k_norm_w;
    void* rope_inv_freq;
    void* final_norm_w;
    void* lm_head_w;
};

class Qwen06BRuntimeParams : public IRuntimeParams {
public:
    void* hidden_states;
    void* position_ids;
    void* logits;
    int newTokens;
    int logitsTokens;
};

class Qwen06BPlatformParams : public IPlatformParams {
public:
    cl_device_id deviceId;
    cl_context context;
    cl_command_queue stream;
};

using ConstantParamsImpl = Qwen06BConstantParams;
using RuntimeParamsImpl = Qwen06BRuntimeParams;
using PlatformParamsImpl = Qwen06BPlatformParams;

}  // namespace mk