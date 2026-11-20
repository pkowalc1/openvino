// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MegaKernel graph transformation for Qwen3-0.6B (28 layers).
//
// The pass:
//  1. Detects the model by counting 56 ReadValue/Assign "past_key_values" pairs.
//  2. Collects per-layer weight Constants (q/k/v/o proj, gate/up/down proj,
//     input_ln, post_attn_ln, q_norm, k_norm) and stacks them along a new dim-0.
//  3. Stacks the 56 ReadValue outputs into two [28,B,8,S,128] tensors via
//     Unsqueeze+Concat (the ReadValue nodes are kept; only wiring changes).
//  4. Creates MegaKernel.
//  5. Replaces the LM-head output with MegaKernel logits.
//  6. Drops the 56 KV-cache Assign sinks: the MegaKernel owns the whole KV cache
//     in its own persistent device buffers.
//
// The original 28-layer SDPA sub-graph is left with no consumers and therefore
// becomes unreachable from Results/Sinks - it is never compiled, so there is no
// duplicate weight memory and no runtime branching (Select) overhead.
//
// Decode includes final normalization and the LM head in one persistent kernel.
// Prefill uses the runtime's batched kernels and vocabulary projection.

#include "../../iOVIntegration.h"
#include "../../ovIntegrationUtils.h"
#include "qwen06BPOCParams.h"

#include "intel_gpu/op/megakernel.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/strided_slice.hpp"
#include "primitive_inst.h"

namespace mk {
namespace {

constexpr int NUM_LAYERS = 28;
constexpr int64_t HIDDEN_SIZE = 1024;
constexpr int64_t VOCAB_SIZE = 151936;
constexpr size_t ROPE_INV_FREQ_ELEMENTS = 64;

const std::vector<std::pair<std::string, std::string>> PROJ_WEIGHTS = {
    {"q_proj",    "self_attn.q_proj.weight"},
    {"k_proj",    "self_attn.k_proj.weight"},
    {"v_proj",    "self_attn.v_proj.weight"},
    {"o_proj",    "self_attn.o_proj.weight"},
    {"gate_proj", "mlp.gate_proj.weight"},
    {"up_proj",   "mlp.up_proj.weight"},
    {"down_proj", "mlp.down_proj.weight"},
};

const std::vector<std::pair<std::string, std::string>> NORM_WEIGHTS = {
    {"input_ln",     "input_layernorm"},
    {"post_attn_ln", "post_attention_layernorm"},
    {"q_norm",       "self_attn.q_norm"},
    {"k_norm",       "self_attn.k_norm"},
};

}  // namespace

bool InsertMegakernelTransformation(const std::shared_ptr<ov::Model>& m) {
    namespace mku = mk::util;
    using ov::intel_gpu::op::MegaKernel;
    using ov::intel_gpu::op::MegaKernelAttrs;
    using ov::intel_gpu::op::MegaKernelDim;

    const ov::NodeVector ops = m->get_ordered_ops();

    if (mku::count_kv_read_values(ops) != 2 * NUM_LAYERS)
        return false;

    // --- 1. Find boundary nodes -------------------------------------------------
    auto embed_gather = mku::find_node(ops, "__module.model.embed_tokens/ov_ext::embedding/Gather");
    OPENVINO_ASSERT(embed_gather, "[MegaKernel] embed_tokens Gather not found");

    auto last_add = mku::find_node(ops, "__module.model.layers.27/aten::add/Add_1");
    OPENVINO_ASSERT(last_add, "[MegaKernel] layer 27 final Add not found");

    auto final_norm = mku::find_node(ops, "__module.model.norm/aten::mul/Multiply_1");
    auto lm_head = mku::find_node(ops, "__module.lm_head/ov_ext::linear/MatMul");
    if (!final_norm || !lm_head)
        return false;
    if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(lm_head)) {
        if (matmul->get_transpose_a() || !matmul->get_transpose_b())
            return false;
    } else if (!ov::is_type<ov::intel_gpu::op::FullyConnected>(lm_head) ||
               !ov::is_type<ov::intel_gpu::op::Placeholder>(lm_head->get_input_node_shared_ptr(2))) {
        return false;
    }
    auto unwrap_convert = [](std::shared_ptr<ov::Node> node) {
        while (ov::is_type<ov::op::v0::Convert>(node))
            node = node->get_input_node_shared_ptr(0);
        return node;
    };
    auto lm_head_weight = ov::as_type_ptr<ov::op::v0::Constant>(unwrap_convert(lm_head->get_input_node_shared_ptr(1)));
    if (!lm_head_weight || lm_head_weight->get_element_type() != ov::element::f16 ||
        lm_head_weight->get_shape() != ov::Shape{VOCAB_SIZE, HIDDEN_SIZE})
        return false;
    std::shared_ptr<ov::op::v0::Constant> final_norm_weight;
    for (const auto& input : final_norm->input_values()) {
        auto weight = ov::as_type_ptr<ov::op::v0::Constant>(unwrap_convert(input.get_node_shared_ptr()));
        if (weight && weight->get_element_type() == ov::element::f16 &&
            (weight->get_shape() == ov::Shape{HIDDEN_SIZE} || weight->get_shape() == ov::Shape{1, 1, HIDDEN_SIZE}))
            final_norm_weight = weight;
    }
    if (!final_norm_weight)
        return false;
    bool last_token_only = false;
    auto head_input = unwrap_convert(lm_head->get_input_node_shared_ptr(0));
    if (auto slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(head_input)) {
        auto begin = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_input_node_shared_ptr(1));
        auto end = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_input_node_shared_ptr(2));
        auto strides = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_input_node_shared_ptr(3));
        if (!begin || !end || !strides || begin->cast_vector<int64_t>() != std::vector<int64_t>{0, -1} ||
            slice->get_begin_mask() != std::vector<int64_t>{1, 0})
            return false;
        const bool forward = strides->cast_vector<int64_t>() == std::vector<int64_t>{1, 1} &&
                             slice->get_end_mask() == std::vector<int64_t>{1, 1};
        const bool reverse = strides->cast_vector<int64_t>() == std::vector<int64_t>{1, -1} &&
                             end->cast_vector<int64_t>() == std::vector<int64_t>{0, -2} &&
                             slice->get_end_mask() == std::vector<int64_t>{1, 0};
        auto has_mask = [](const std::vector<int64_t>& mask) {
            return std::any_of(mask.begin(), mask.end(), [](int64_t value) { return value != 0; });
        };
        if ((!forward && !reverse) || has_mask(slice->get_new_axis_mask()) ||
            has_mask(slice->get_shrink_axis_mask()) || has_mask(slice->get_ellipsis_mask()))
            return false;
        head_input = unwrap_convert(slice->get_input_node_shared_ptr(0));
        last_token_only = true;
    }
    if (head_input != final_norm)
        return false;

    std::shared_ptr<ov::Node> pos_ids_node = nullptr, beam_idx_node = nullptr;
    for (auto& op : ops) {
        if (!ov::is_type<ov::op::v0::Parameter>(op)) continue;
        const auto& name = op->get_friendly_name();
        if (name == "position_ids") pos_ids_node = op;
        if (name == "beam_idx")     beam_idx_node = op;
    }
    OPENVINO_ASSERT(pos_ids_node && beam_idx_node, "[MegaKernel] Missing Parameter nodes");
    // Runs after ConvertPrecision, so this is the dtype the runtime will receive.
    OPENVINO_ASSERT(pos_ids_node->get_output_element_type(0) == ov::element::i64,
                    "[MegaKernel] supports only i64 position_ids");

    // --- 2. Stacked projection weights ------------------------------------------
    std::map<std::string, std::shared_ptr<ov::op::v0::Constant>> stacked_proj;
    for (auto& [tag, suffix] : PROJ_WEIGHTS) {
        std::vector<std::shared_ptr<ov::op::v0::Constant>> per_layer;
        per_layer.reserve(NUM_LAYERS);
        for (int l = 0; l < NUM_LAYERS; ++l)
            per_layer.push_back(mku::get_proj_weight(ops, l, suffix));
        stacked_proj[tag] = mku::stack_constants(per_layer, "megakernel_stacked_" + tag);
    }

    // --- 3. Stacked norm weights -------------------------------------------------
    std::map<std::string, std::shared_ptr<ov::op::v0::Constant>> stacked_norm;
    for (auto& [tag, fragment] : NORM_WEIGHTS) {
        std::vector<std::shared_ptr<ov::op::v0::Constant>> per_layer;
        per_layer.reserve(NUM_LAYERS);
        for (int l = 0; l < NUM_LAYERS; ++l)
            per_layer.push_back(mku::squeeze_leading_ones(mku::get_norm_weight(ops, l, fragment)));
        stacked_norm[tag] = mku::stack_constants(per_layer, "megakernel_stacked_" + tag);
    }

    // --- 4. rope_inv_freq --------------------------------------------------------
    auto rope_inv_freq = mku::find_rope_inv_freq(ops, ROPE_INV_FREQ_ELEMENTS);

    // --- 5. Stack KV caches ------------------------------------------------------
    std::vector<std::shared_ptr<ov::op::v6::Assign>> key_as, val_as;
    mku::collect_sorted_assign(ops, NUM_LAYERS, key_as, val_as);

    std::vector<std::shared_ptr<ov::op::v6::ReadValue>> key_rvs, val_rvs;
    mku::collect_sorted_rv(ops, NUM_LAYERS, key_rvs, val_rvs);
    auto past_key = mku::build_stacked_kv(key_rvs);
    auto past_val = mku::build_stacked_kv(val_rvs);

    // --- 6. Build MegaKernel -----------------------------------------------------
    MegaKernelAttrs attrs;
    attrs.outputs = {{ov::element::f32, {MegaKernelDim::of_input(0, 0),
                                       last_token_only ? MegaKernelDim::constant(1) : MegaKernelDim::of_input(0, 1),
                                       MegaKernelDim::constant(VOCAB_SIZE)}}};

    ov::OutputVector mk_inputs = {
        embed_gather->output(0),                 // 0  hidden_states
        pos_ids_node->output(0),                 // 1  position_ids
        beam_idx_node->output(0),                // 2  beam_idx
        past_key,                                // 3  past_key  [28,B,8,S,128]
        past_val,                                // 4  past_val
        stacked_proj["q_proj"]->output(0),       // 5
        stacked_proj["k_proj"]->output(0),       // 6
        stacked_proj["v_proj"]->output(0),       // 7
        stacked_proj["o_proj"]->output(0),       // 8
        stacked_proj["gate_proj"]->output(0),    // 9
        stacked_proj["up_proj"]->output(0),      // 10
        stacked_proj["down_proj"]->output(0),    // 11
        stacked_norm["input_ln"]->output(0),     // 12
        stacked_norm["post_attn_ln"]->output(0), // 13
        stacked_norm["q_norm"]->output(0),       // 14
        stacked_norm["k_norm"]->output(0),       // 15
        rope_inv_freq,                           // 16
        final_norm_weight->output(0),
        lm_head_weight->output(0),
    };

    auto mk_node = std::make_shared<MegaKernel>(mk_inputs, attrs);
    mk_node->set_friendly_name("MegaKernel");
    ov::copy_runtime_info(lm_head, mk_node);

    ov::Output<ov::Node> logits = mk_node->output(0);
    if (logits.get_element_type() != lm_head->get_output_element_type(0)) {
        auto convert = std::make_shared<ov::op::v0::Convert>(logits, lm_head->get_output_element_type(0));
        logits = convert->output(0);
    }
    logits.set_names(lm_head->output(0).get_names());
    auto logits_consumers = lm_head->output(0).get_target_inputs();
    for (auto& input : logits_consumers)
        input.replace_source_output(logits);

    // --- 8. Drop the KV-cache Assign sinks ---------------------------------------
    // The MegaKernel keeps the entire KV cache in its own persistent device buffers
    // (see MegaKernelFastImpl), so OpenVINO's per-layer KV Variables are redundant.
    // Removing the Assign sinks eliminates 56 Reorder/Convert/Assign ops that
    // otherwise run every decode step purely to feed unused state.
    for (auto& a : key_as)
        m->remove_sink(a);
    for (auto& a : val_as)
        m->remove_sink(a);

    m->validate_nodes_and_infer_types();

    mku::verify_and_dump(m, 1, 0);
    return true;
}

static void* usm_raw(cldnn::memory& m) {
    OPENVINO_ASSERT(cldnn::memory_capabilities::is_usm_type(m.get_allocation_type()),
                    "[MegaKernel] port memory must be a USM allocation for the task-system path");
    return m.buffer_ptr();
}


std::shared_ptr<const mk::IConstantParams> GetMegaKernelConstantParams(cldnn::primitive_inst* prim) {
    auto constant_params = std::make_shared<Qwen06BConstantParams>();
    constant_params->q_proj_w =usm_raw(prim->input_memory(5));
    constant_params->k_proj_w = usm_raw(prim->input_memory(6));
    constant_params->v_proj_w = usm_raw(prim->input_memory(7));
    constant_params->o_proj_w = usm_raw(prim->input_memory(8));
    constant_params->gate_proj_w = usm_raw(prim->input_memory(9));
    constant_params->up_proj_w = usm_raw(prim->input_memory(10));
    constant_params->down_proj_w = usm_raw(prim->input_memory(11));
    constant_params->input_ln_w = usm_raw(prim->input_memory(12));
    constant_params->post_attn_ln_w = usm_raw(prim->input_memory(13));
    constant_params->q_norm_w = usm_raw(prim->input_memory(14));
    constant_params->k_norm_w = usm_raw(prim->input_memory(15));
    constant_params->rope_inv_freq = usm_raw(prim->input_memory(16));
    constant_params->final_norm_w = usm_raw(prim->input_memory(17));
    constant_params->lm_head_w = usm_raw(prim->input_memory(18));
    return constant_params;
}

std::shared_ptr<const mk::IRuntimeParams> GetMegaKernelRuntimeParams(cldnn::primitive_inst* prim) {
    auto runtime_params = std::make_shared<Qwen06BRuntimeParams>();
    runtime_params->hidden_states = usm_raw(prim->input_memory(0));
    runtime_params->position_ids = usm_raw(prim->input_memory(1));
    runtime_params->logits = usm_raw(prim->output_memory(0));
    runtime_params->newTokens = static_cast<int>(prim->get_impl_params()->get_input_layout(0).get_shape().at(1));
    runtime_params->logitsTokens = static_cast<int>(prim->output_memory(0).get_layout().get_shape().at(1));
    OPENVINO_ASSERT(prim->get_impl_params()->get_input_layout(0).get_shape().at(0) == 1,
                    "[MegaKernel] supports only batch size 1");
    return runtime_params;
}

}  // namespace mk
