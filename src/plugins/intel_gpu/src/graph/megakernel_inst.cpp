// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "megakernel_inst.h"

#include "intel_gpu/op/megakernel.hpp"
#include "json_object.h"
#include "primitive_type_base.h"

#include <sstream>

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(megakernel)

// ---------------------------------------------------------------------------
// Shape inference
// ---------------------------------------------------------------------------
// Outputs:
//   [0]  hidden_states_out  [B, S, hidden_size]
//   [1]  present_key        [num_layers, B, num_kv_heads, S_past + S, head_dim]
//   [2]  present_val        [num_layers, B, num_kv_heads, S_past + S, head_dim]

template <typename ShapeType>
std::vector<layout> megakernel_inst::calc_output_layouts(megakernel_node const& /*node*/,
                                                                 const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<megakernel>();
    const auto& in_layout = impl_param.get_input_layout(0);

    auto ps = ov::intel_gpu::op::megakernel_output_shape(
        in_layout.get<ShapeType>(),
        impl_param.input_layouts.size() > 1 ? impl_param.get_input_layout(1).get<ShapeType>() : ShapeType{},
        desc->hidden_size, desc->num_heads, desc->head_dim);

    auto dt = data_types::f32;
    if (!desc->output_data_types.empty() && desc->output_data_types[0].has_value())
        dt = desc->output_data_types[0].value();
    std::vector<layout> outs;
    outs.emplace_back(ps, dt, in_layout.format);
    return outs;
}

layout megakernel_inst::calc_output_layout(megakernel_node const& node,
                                                   kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template std::vector<layout> megakernel_inst::calc_output_layouts<ov::PartialShape>(
    megakernel_node const& node,
    const kernel_impl_params& impl_param);

// ---------------------------------------------------------------------------
// to_string / constructor
// ---------------------------------------------------------------------------
std::string megakernel_inst::to_string(megakernel_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    std::stringstream ss;
    json_composite info;
    info.add("num_layers",   desc->num_layers);
    info.add("hidden_size",  desc->hidden_size);
    info.add("num_kv_heads", desc->num_kv_heads);
    info.add("head_dim",     desc->head_dim);
    node_info->add("megakernel_info", info);
    node_info->dump(ss);
    return ss.str();
}

megakernel_inst::typed_primitive_inst(network& network, megakernel_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
