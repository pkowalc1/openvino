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
};

class Qwen06BRuntimeParams : public IRuntimeParams {
public:
    void* hidden_states;
    void* position_ids;
    void* hidden_states_out;
    int newTokens;

    // Hand-over of a KV cache produced outside the megakernel (the OpenVINO
    // prefill path). When import_past is set, the runtime copies past_len tokens
    // of every layer/head into its own cache before running the decode step.
    // Each array holds one entry per layer; a layer's buffer is laid out as
    // [num_kv_heads, <its own> seq stride, head_dim] halfs.
    const void* const* past_key = nullptr;
    const void* const* past_value = nullptr;
    const int* past_key_stride = nullptr;
    const int* past_value_stride = nullptr;
    int past_len = 0;
    bool import_past = false;
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