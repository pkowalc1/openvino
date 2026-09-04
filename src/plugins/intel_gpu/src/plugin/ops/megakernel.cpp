// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/megakernel.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/megakernel.hpp"

// Alias into the ov::op::internal namespace so REGISTER_FACTORY_IMPL can find it.
namespace ov {
namespace op {
namespace internal {
using MegaKernel = ov::intel_gpu::op::MegaKernel;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateMegaKernelOp(ProgramBuilder& p,
                                      const std::shared_ptr<ov::intel_gpu::op::MegaKernel>& op) {
    validate_inputs_count(op, {15});
    auto inputs = p.GetInputInfo(op);
    const auto& attrs = op->get_attrs();

    std::vector<cldnn::input_info> prim_inputs;
    prim_inputs.reserve(inputs.size());
    for (auto& ii : inputs)
        prim_inputs.push_back(cldnn::input_info(ii));

    auto prim = cldnn::megakernel(
        layer_type_name_ID(op),
        prim_inputs,
        attrs.num_layers,
        attrs.hidden_size,
        attrs.num_kv_heads,
        attrs.head_dim,
        attrs.num_attention_heads,
        attrs.intermediate_size,
        attrs.rms_norm_eps);

    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, MegaKernel);

}  // namespace ov::intel_gpu
