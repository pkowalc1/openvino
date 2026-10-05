// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/megakernel.hpp"

namespace ov::intel_gpu::op {

MegaKernel::MegaKernel(const ov::OutputVector& inputs, const MegaKernelAttrs& attrs)
    : Op(inputs),
      m_attrs(attrs) {
    validate_and_infer_types();
}

bool MegaKernel::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("num_layers",          m_attrs.num_layers);
    visitor.on_attribute("hidden_size",         m_attrs.hidden_size);
    visitor.on_attribute("num_attention_heads", m_attrs.num_attention_heads);
    visitor.on_attribute("num_kv_heads",        m_attrs.num_kv_heads);
    visitor.on_attribute("head_dim",            m_attrs.head_dim);
    visitor.on_attribute("intermediate_size",   m_attrs.intermediate_size);
    visitor.on_attribute("rms_norm_eps",        m_attrs.rms_norm_eps);
    visitor.on_attribute("kind",                m_attrs.kind);
    visitor.on_attribute("out_type",            m_attrs.out_type);
    return true;
}

void MegaKernel::validate_and_infer_types() {
    // Shape follows port 0 with the last dimension replaced by hidden_size, so the
    // same op covers a shape-preserving decoder block and a projection such as
    // lm_head that widens the trailing dimension.
    auto ps = megakernel_output_shape(get_input_partial_shape(0),
                                      get_input_size() > 1 ? get_input_partial_shape(1) : ov::PartialShape{},
                                      m_attrs.hidden_size, m_attrs.num_attention_heads, m_attrs.head_dim);
    set_output_type(0, m_attrs.out_type, ps);
}

std::shared_ptr<ov::Node> MegaKernel::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<MegaKernel>(new_args, m_attrs);
}

}  // namespace ov::intel_gpu::op
