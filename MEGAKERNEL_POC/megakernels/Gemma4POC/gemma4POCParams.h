#pragma once
#include <CL/cl.h>

#include "../iMegakernelRuntime.h"

namespace mk {

// Which subgraph a MegaKernel op replaces (MegaKernelAttrs::kind).
enum Gemma4Kind : int {
    kLmHead = 0,  // opt-in INT8 lm_head GEMV
    kFfn = 1,     // post-attention residual .. layer_scalar (dense MLP + MoE)
    kAttnIn = 2,  // input_layernorm .. q/k/v projections, norms and RoPE
    kMaskFull = 3,     // attention mask of the full-attention layers
    kMaskSliding = 4,  // attention mask of the sliding-window layers
};

// Shapes of the Gemma-4-26B-A4B FFN block; the transformation refuses layers
// that do not match.
constexpr int kHidden = 2816;
constexpr int kInter = 2112;
constexpr int kExpertInter = 704;
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kQuantGroup = 64;

// Graph port order of a kFfn MegaKernel, fixed by gemma4POCTransformation.cpp.
// Int4 matrices arrive repacked into sub-group records (gemma4Repack.h).
enum FfnPort : int {
    kFfnO = 0,          // f16 [B,S,KA] attention output (o_proj input)
    kFfnH,              // f16 [B,S,H]  residual entering the layer
    kFfnWPostAttn,      // f16 [H]      post_attention_layernorm
    kFfnWPreFf,         // f16 [H]      pre_feedforward_layernorm
    kFfnGate,           // u8 records   mlp.gate_proj [I,H]
    kFfnUp,             // u8 records   mlp.up_proj   [I,H]
    kFfnDown,           // u8 records   mlp.down_proj [H,I]
    kFfnWPostFf1,       // f16 [H]      post_feedforward_layernorm_1
    kFfnWRouter,        // f16 [H]      gamma of the RMS feeding router.proj
    kFfnRouterW,        // f16 [E,H]
    kFfnWPreFf2,        // f16 [H]      pre_feedforward_layernorm_2
    kFfnEGate,          // u8 records   experts gate [E*IE,H]
    kFfnEUp,            // u8 records   experts up   [E*IE,H]
    kFfnEDown,          // u8 records   experts down [E*H,IE]
    kFfnWPostFf2,       // f16 [H]      post_feedforward_layernorm_2
    kFfnWPostFf,        // f16 [H]      post_feedforward_layernorm
    kFfnLayerScalar,    // f16 [1]
    kFfnOProj,          // u8 records   self_attn.o_proj [H,KA]
    kFfnPorts
};

// Graph port order of a kAttnIn MegaKernel. Its output is f16 [B,S,(HQ+2*HK)*D]
// holding the roped q, the roped k and the normalised v of each token.
enum AttnPort : int {
    kAttnX = 0,         // f16 [B,S,H]   residual entering the layer
    kAttnPos,           // i32/i64 [B,S] position_ids
    kAttnWIn,           // f16 [H]       input_layernorm
    kAttnQkv,           // u8 records    q|k|v projection rows [R,H]
    kAttnWQ,            // f16 [D]       q_norm
    kAttnWK,            // f16 [D]       k_norm
    kAttnMeta,          // i32 [16]      see attn_post in gemma4FfnKernels.h
    kAttnInvFreq,       // f32 [C/2]     rotary_emb inv_freq
    kAttnPorts
};
constexpr int kAttnMaxHeads = 64;
constexpr int kAttnMaxRows = 16384;

// Graph port order of a kMaskFull / kMaskSliding MegaKernel; output [B,1,S,L].
enum MaskPort : int {
    kMaskAm = 0,        // i32/i64 [B,L] attention_mask
    kMaskTti,           // i32/i64 [B,S] token_type_ids
    kMaskMeta,          // f32 [16]      see mask_fill in gemma4FfnKernels.h
    kMaskPorts
};

class Gemma4ConstantParams : public IConstantParams {
public:
    int kind = kLmHead;

    // kLmHead
    void* lm_head_q = nullptr;      // u8  [N, K]
    void* lm_head_zp = nullptr;     // u8  [N]
    void* lm_head_scale = nullptr;  // f16 [N]
    int N = 0;
    int K = 0;

    // kFfn, indexed by FfnPort; activation ports stay null.
    void* ffn[kFfnPorts] = {};

    // kAttnIn, indexed by AttnPort; activation ports stay null.
    void* attn[kAttnPorts] = {};
    int attn_rows = 0;  // rows of the kAttnQkv matrix

    void* mask_meta = nullptr;  // kMask*
};

class Gemma4RuntimeParams : public IRuntimeParams {
public:
    void* in = nullptr;    // kLmHead: f16 [tokens, K]   kFfn: attention output [tokens, ka]   kAttnIn: x
    void* in2 = nullptr;   // kFfn: residual   kAttnIn: position_ids
    void* in3 = nullptr;
    void* out = nullptr;   // kLmHead: f32 [tokens, N]   kFfn: f16 [tokens, H]
    int tokens = 0;        // kMask*: batch
    int ka = 0;            // kFfn: o_proj input width   kAttnIn: RoPE table width   kMask*: L
    int s = 0;             // kMask*: query length
};

class Gemma4PlatformParams : public IPlatformParams {
public:
    cl_device_id deviceId;
    cl_context context;
    cl_command_queue stream;
};

using ConstantParamsImpl = Gemma4ConstantParams;
using RuntimeParamsImpl = Gemma4RuntimeParams;
using PlatformParamsImpl = Gemma4PlatformParams;

}  // namespace mk
