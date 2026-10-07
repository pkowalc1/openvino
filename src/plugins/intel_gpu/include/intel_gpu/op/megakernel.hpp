// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

struct MegaKernelAttrs {
    int64_t num_layers           = 28;
    int64_t hidden_size          = 1024;
    int64_t num_attention_heads  = 16;
    int64_t num_kv_heads         = 8;
    int64_t head_dim             = 128;
    int64_t intermediate_size    = 3072;
    float   rms_norm_eps         = 1e-6f;
    // Selects the sub-kernel inside the megakernel library, for libraries that
    // replace more than one kind of subgraph.
    int64_t kind                 = 0;
    ov::element::Type out_type   = ov::element::f32;
};

// hidden_size value selecting an attention-mask output [B, 1, S, L] built from
// port 0 [B, L] (attention_mask) and port 1 [B, S] (any per-query tensor).
constexpr int64_t kMegaKernelMaskShape = -1;
// hidden_size value selecting a head-major output [B, heads, S, head_dim] from
// port 0 [B, S, ...]; heads and head_dim are passed in the same call.
constexpr int64_t kMegaKernelHeadsShape = -2;

template <typename PS>
PS megakernel_output_shape(const PS& in0, const PS& in1, int64_t hidden_size, int64_t heads = 0, int64_t head_dim = 0) {
    PS ps = in0;
    if (hidden_size == kMegaKernelHeadsShape) {
        if (in0.rank().is_static() && in0.size() >= 2)
            return PS{in0[0], heads, in0[1], head_dim};
        return PS::dynamic(4);
    }
    if (hidden_size == kMegaKernelMaskShape) {
        if (in0.rank().is_static() && in0.size() == 2 && in1.rank().is_static() && in1.size() == 2)
            return PS{in0[0], 1, in1[1], in0[1]};
        return PS::dynamic(4);
    }
    if (ps.rank().is_static() && ps.rank().get_length() > 0 && hidden_size > 0)
        ps[ps.rank().get_length() - 1] = hidden_size;
    return ps;
}

class MegaKernel : public ov::op::Op {
public:
    OPENVINO_OP("MegaKernel", "gpu_opset");

    MegaKernel() = default;

    MegaKernel(const ov::OutputVector& inputs, const MegaKernelAttrs& attrs);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const MegaKernelAttrs& get_attrs() const { return m_attrs; }

private:
    MegaKernelAttrs m_attrs;
};

}  // namespace ov::intel_gpu::op
