// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MegaKernel graph transformation for Qwen3-0.6B (28 layers).
//
// The pass (MegaKernel decode model of the two-model PoC setup):
//  1. Detects the model by counting 56 ReadValue/Assign "past_key_values" pairs.
//  2. Collects per-layer weight Constants (q/k/v/o proj, gate/up/down proj,
//     input_ln, post_attn_ln, q_norm, k_norm) and stacks them along a new dim-0.
//  3. Stacks the 56 ReadValue outputs into two [28,B,8,S,128] tensors via
//     Unsqueeze+Concat (the ReadValue nodes are kept; only wiring changes).
//  4. Creates MegaKernel.
//  5. Rewires model.norm to consume the MegaKernel hidden-state output directly
//     (a single Convert to f16 to match the downstream precision).
//  6. Drops the 56 KV-cache Assign sinks: the MegaKernel owns the whole KV cache
//     in its own persistent device buffers.
//
// The original 28-layer SDPA sub-graph is left with no consumers and therefore
// becomes unreachable from Results/Sinks - it is never compiled, so there is no
// duplicate weight memory and no runtime branching (Select) overhead.
//
// Two-model PoC: this pass produces the DECODE model. Prefill is served by a
// separate, unmodified OpenVINO model (compile with OV_MEGAKERNEL_DISABLE=1) so
// the MegaKernel never handles prefill and only decode latency is measured.

#include "../../iMegakernelTransformation.h"
#include "../../megakernelTransformationUtils.h"

#include "intel_gpu/op/megakernel.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"

namespace mk {
namespace {

constexpr int NUM_LAYERS = 28;
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

    const ov::NodeVector ops = m->get_ordered_ops();

    if (mku::count_kv_read_values(ops) != 2 * NUM_LAYERS)
        return false;

    // --- 1. Find boundary nodes -------------------------------------------------
    auto embed_gather = mku::find_node(ops, "__module.model.embed_tokens/ov_ext::embedding/Gather");
    OPENVINO_ASSERT(embed_gather, "[MegaKernel] embed_tokens Gather not found");

    auto last_add = mku::find_node(ops, "__module.model.layers.27/aten::add/Add_1");
    OPENVINO_ASSERT(last_add, "[MegaKernel] layer 27 final Add not found");

    std::shared_ptr<ov::Node> pos_ids_node = nullptr, beam_idx_node = nullptr;
    for (auto& op : ops) {
        if (!ov::is_type<ov::op::v0::Parameter>(op)) continue;
        const auto& name = op->get_friendly_name();
        if (name == "position_ids") pos_ids_node = op;
        if (name == "beam_idx")     beam_idx_node = op;
    }
    OPENVINO_ASSERT(pos_ids_node && beam_idx_node, "[MegaKernel] Missing Parameter nodes");

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
    MegaKernelAttrs attrs;  // defaults already describe Qwen3-0.6B

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
    };

    auto mk_node = std::make_shared<MegaKernel>(mk_inputs, attrs);
    mk_node->set_friendly_name("MegaKernel");
    ov::copy_runtime_info(last_add, mk_node);

    // --- 7. Rewire model.norm <- MegaKernel hidden-state output -------------------
    auto mk_hidden_f16 = std::make_shared<ov::op::v0::Convert>(mk_node->output(0), ov::element::f16);
    mk_hidden_f16->set_friendly_name("mk_hidden_f16");
    auto last_add_consumers = last_add->output(0).get_target_inputs();
    for (auto& inp : last_add_consumers)
        inp.replace_source_output(mk_hidden_f16->output(0));

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

    mku::verify_and_dump(m, 2 * NUM_LAYERS);
    return true;
}

}  // namespace mk
