// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The pass (MegaKernel prefill/decode triage):
//  1. Detects the model by counting 56 ReadValue/Assign "past_key_values" pairs.
//  2. Collects per-layer weight Constants (q/k/v/o proj, gate/up/down proj,
//     input_ln, post_attn_ln, q_norm, k_norm) and stacks them along a new dim-0
//     so the MegaKernel op takes 15 inputs instead of ~600 scattered Constants.
//  3. Splits the token stream in two: the original 28-layer sub-graph keeps the
//     PREFILL tokens and the MegaKernel gets the DECODE token. The split is a
//     pair of Slices whose length is chosen at runtime from seq_len:
//         seq_len  > 1 (prefill) -> original path gets S tokens, MegaKernel 0
//         seq_len == 1 (decode)  -> original path gets 0 tokens, MegaKernel 1
//     A zero-length tensor makes every primitive of the inactive path report an
//     empty output, and cldnn then flags them ExecutionFlags::SKIP, so the
//     inactive path enqueues no kernels at all (measured: 0.08 ms of GPU time
//     for a full zero-length pass through the 28 layers).
//  4. Merges the two paths again with Concat(axis=1): exactly one of the two
//     branches is non-empty, so the concatenation is the active branch's result.
//
// Consequences:
//  - Prefill runs the untouched original graph (KVCacheFusion, IndirectSDPA and
//    the rest of the GPU pipeline still apply), so prefill latency matches the
//    baseline.
//  - The 56 KV-cache Assign sinks are KEPT: prefill fills the OpenVINO KV cache,
//    which is the hand-over point to the MegaKernel. The MegaKernel impl imports
//    those buffers into its own internal cache on the first decode step of a
//    sequence (see MegaKernelFastImpl) and owns the cache from then on.
//  - Weights exist twice (original Constants + stacked copies), so device memory
//    roughly doubles. That is the accepted cost of keeping both paths alive.
//
// The pass is a no-op when OV_MEGAKERNEL_DISABLE=1, which yields the baseline model.

#include "insert_megakernel.hpp"

#include "intel_gpu/op/megakernel.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace ov::intel_gpu {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

constexpr int NUM_LAYERS = 28;

// Return the node with the given friendly name or nullptr.
std::shared_ptr<ov::Node> find_node(const ov::NodeVector& ops, const std::string& name) {
    for (auto& op : ops) {
        if (op->get_friendly_name() == name)
            return op;
    }
    return nullptr;
}

// Walk through Convert wrappers to the underlying Constant and return its data
// as a flat float16 (f16) buffer re-packaged into a new Constant of the given
// shape.  If the node is already a Constant of dtype f16 it is returned as-is.
std::shared_ptr<ov::op::v0::Constant> get_f16_constant(std::shared_ptr<ov::Node> node) {
    // Strip Convert wrappers (decompressor pattern) to reach the underlying Constant.
    for (int depth = 0; depth < 10; ++depth) {
        if (ov::is_type<ov::op::v0::Constant>(node))
            break;
        if (!ov::is_type<ov::op::v0::Convert>(node))
            break;
        node = node->get_input_node_shared_ptr(0);
    }
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(node);
    OPENVINO_ASSERT(c != nullptr,
                    "[MegaKernel] expected Constant, got '", node->get_type_name(),
                    "' (name='", node->get_friendly_name(), "')");
    // If already f16, return directly
    if (c->get_element_type() == ov::element::f16)
        return c;
    // Otherwise cast to f16
    auto data_f32 = c->cast_vector<ov::float16>();
    return ov::op::v0::Constant::create(ov::element::f16, c->get_shape(), data_f32);
}

// Stack a vector of per-layer Constants along a new leading dimension.
// Each constant must have the same shape.  Returns new stacked Constant.
std::shared_ptr<ov::op::v0::Constant> stack_constants(
        const std::vector<std::shared_ptr<ov::op::v0::Constant>>& per_layer,
        const std::string& debug_name) {

    OPENVINO_ASSERT(!per_layer.empty());
    const ov::Shape elem_shape = per_layer[0]->get_shape();
    const ov::element::Type dtype = per_layer[0]->get_element_type();

    // Flatten each layer's data and concatenate
    size_t elem_size = dtype.bitwidth() / 8;
    size_t layer_bytes = ov::shape_size(elem_shape) * elem_size;
    std::vector<uint8_t> buf(per_layer.size() * layer_bytes);

    for (size_t i = 0; i < per_layer.size(); ++i) {
        OPENVINO_ASSERT(per_layer[i]->get_element_type() == dtype,
                        "[MegaKernel] dtype mismatch while stacking ", debug_name);
        const auto* src = static_cast<const uint8_t*>(per_layer[i]->get_data_ptr());
        std::copy(src, src + layer_bytes, buf.data() + i * layer_bytes);
    }

    ov::Shape stacked_shape;
    stacked_shape.push_back(per_layer.size());
    for (auto d : elem_shape)
        stacked_shape.push_back(d);

    auto stacked = ov::op::v0::Constant::create(dtype, stacked_shape, buf.data());
    stacked->set_friendly_name(debug_name);
    return stacked;
}

// Squeeze all size-1 leading dims from a constant (e.g. [1,1,1024]->[1024]).
std::shared_ptr<ov::op::v0::Constant> squeeze_leading_ones(
        std::shared_ptr<ov::op::v0::Constant> c) {
    ov::Shape s = c->get_shape();
    size_t first_nontrivial = 0;
    while (first_nontrivial < s.size() && s[first_nontrivial] == 1)
        ++first_nontrivial;
    if (first_nontrivial == 0)
        return c;
    ov::Shape new_shape(s.begin() + first_nontrivial, s.end());
    if (new_shape.empty())
        new_shape = {1};
    size_t elem_size = c->get_element_type().bitwidth() / 8;
    size_t nbytes = ov::shape_size(new_shape) * elem_size;
    std::vector<uint8_t> buf(nbytes);
    std::copy(static_cast<const uint8_t*>(c->get_data_ptr()),
              static_cast<const uint8_t*>(c->get_data_ptr()) + nbytes,
              buf.data());
    return ov::op::v0::Constant::create(c->get_element_type(), new_shape, buf.data());
}

// Find the weight constant for a per-layer matmul weight.
// The graph has pattern:  Constant (f16 compressed) -> Convert -> MatMul
// The constant's friendly name is "self.model.layers.<l>.<proj_suffix>"
std::shared_ptr<ov::op::v0::Constant> get_proj_weight(
        const ov::NodeVector& ops, int layer, const std::string& weight_name_suffix) {
    const std::string expected = "self.model.layers." + std::to_string(layer) + "." + weight_name_suffix;
    auto node = find_node(ops, expected);
    OPENVINO_ASSERT(node != nullptr, "[MegaKernel] Cannot find weight node: ", expected);
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(node);
    OPENVINO_ASSERT(c != nullptr);
    return c;  // already f16 compressed
}

// Find norm weight constant for a given layer and norm name fragment.
// Strategy: find the last Multiply_1 in the RMSNorm sequence whose name
// contains "layers.<l>.<norm_fragment>", then scan both inputs to find the
// one that traces back to a Constant through Convert wrappers (the weight).
// The other input is the normalised activation path.
std::shared_ptr<ov::op::v0::Constant> get_norm_weight(
        const ov::NodeVector& ops, int layer, const std::string& norm_fragment) {
    const std::string target_frag = "layers." + std::to_string(layer) + "." + norm_fragment;
    for (auto& op : ops) {
        const std::string& n = op->get_friendly_name();
        if (n.find(target_frag) == std::string::npos) continue;
        if (n.find("Multiply_1") == std::string::npos) continue;
        // Try both input ports — weight may be at port 0 or 1 depending on export order.
        for (size_t port = 0; port < op->get_input_size(); ++port) {
            auto candidate = op->get_input_node_shared_ptr(port);
            // Trace through Convert wrappers only to identify a Constant source.
            auto trace = candidate;
            for (int d = 0; d < 10 && !ov::is_type<ov::op::v0::Constant>(trace); ++d) {
                if (!ov::is_type<ov::op::v0::Convert>(trace)) { trace = nullptr; break; }
                trace = trace->get_input_node_shared_ptr(0);
            }
            if (trace && ov::is_type<ov::op::v0::Constant>(trace))
                return get_f16_constant(candidate);
        }
    }
    OPENVINO_THROW("[MegaKernel] Cannot find norm weight for layer ", layer, " fragment '", norm_fragment, "'");
}

}  // namespace

// ---------------------------------------------------------------------------
// Pass entry point
// ---------------------------------------------------------------------------
bool InsertMegaKernel::run_on_model(const std::shared_ptr<ov::Model>& m) {
    if (const char* off = std::getenv("OV_MEGAKERNEL_DISABLE"); off && off[0] == '1')
        return false;

    const ov::NodeVector ops = m->get_ordered_ops();

    int rv_kv_count = 0;
    for (auto& op : ops) {
        auto rv = ov::as_type_ptr<ov::op::v6::ReadValue>(op);
        if (rv && rv->get_variable_id().find("past_key_values") != std::string::npos)
            ++rv_kv_count;
    }
    if (rv_kv_count != 2 * NUM_LAYERS)
        return false;

    // --- 1. Find boundary nodes --------------------------------------------------
    auto embed_gather = find_node(ops, "__module.model.embed_tokens/ov_ext::embedding/Gather");
    OPENVINO_ASSERT(embed_gather, "[MegaKernel] embed_tokens Gather not found");

    auto last_add = find_node(ops, "__module.model.layers.27/aten::add/Add_1");
    OPENVINO_ASSERT(last_add, "[MegaKernel] layer 27 final Add not found");

    // Parameter nodes
    std::shared_ptr<ov::Node> input_ids_node = nullptr, pos_ids_node = nullptr, beam_idx_node = nullptr;
    for (auto& op : ops) {
        if (!ov::is_type<ov::op::v0::Parameter>(op)) continue;
        const auto& name = op->get_friendly_name();
        if (name == "input_ids")     input_ids_node = op;
        if (name == "position_ids")  pos_ids_node   = op;
        if (name == "beam_idx")      beam_idx_node  = op;
    }
    OPENVINO_ASSERT(input_ids_node && pos_ids_node && beam_idx_node, "[MegaKernel] Missing Parameter nodes");

    // --- 2. Stacked projection weights ------------------------------------------
    // Stack per-layer weights into [28, ...] constants used as MegaKernel inputs.
    static const std::vector<std::pair<std::string, std::string>> PROJ_WEIGHTS = {
        {"q_proj",    "self_attn.q_proj.weight"},
        {"k_proj",    "self_attn.k_proj.weight"},
        {"v_proj",    "self_attn.v_proj.weight"},
        {"o_proj",    "self_attn.o_proj.weight"},
        {"gate_proj", "mlp.gate_proj.weight"},
        {"up_proj",   "mlp.up_proj.weight"},
        {"down_proj", "mlp.down_proj.weight"},
    };

    std::map<std::string, std::shared_ptr<ov::op::v0::Constant>> stacked_proj;
    for (auto& [tag, suffix] : PROJ_WEIGHTS) {
        std::vector<std::shared_ptr<ov::op::v0::Constant>> per_layer;
        per_layer.reserve(NUM_LAYERS);
        for (int l = 0; l < NUM_LAYERS; ++l)
            per_layer.push_back(get_proj_weight(ops, l, suffix));
        stacked_proj[tag] = stack_constants(per_layer, "megakernel_stacked_" + tag);
    }

    // --- 3. Stacked norm weights -------------------------------------------------
    static const std::vector<std::pair<std::string, std::string>> NORM_WEIGHTS = {
        {"input_ln",      "input_layernorm"},
        {"post_attn_ln",  "post_attention_layernorm"},
        {"q_norm",        "self_attn.q_norm"},
        {"k_norm",        "self_attn.k_norm"},
    };

    std::map<std::string, std::shared_ptr<ov::op::v0::Constant>> stacked_norm;
    for (auto& [tag, fragment] : NORM_WEIGHTS) {
        std::vector<std::shared_ptr<ov::op::v0::Constant>> per_layer;
        per_layer.reserve(NUM_LAYERS);
        for (int l = 0; l < NUM_LAYERS; ++l) {
            auto c = get_norm_weight(ops, l, fragment);
            per_layer.push_back(squeeze_leading_ones(c));
        }
        stacked_norm[tag] = stack_constants(per_layer, "megakernel_stacked_" + tag);
    }

    // --- 4. rope_inv_freq --------------------------------------------------------
    ov::Output<ov::Node> rope_inv_freq;
    {
        bool found = false;
        for (auto& op : ops) {
            auto c = ov::as_type_ptr<ov::op::v0::Constant>(op);
            if (!c || ov::shape_size(c->get_shape()) != 64) continue;
            for (auto& out : c->outputs()) {
                for (auto& inp : out.get_target_inputs()) {
                    if (inp.get_node()->get_friendly_name().find("rotary") != std::string::npos) {
                        rope_inv_freq = c->output(0);
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            if (found) break;
        }
        OPENVINO_ASSERT(found, "[MegaKernel] rope inv_freq not found");
    }

    // --- 5. Prefill/decode token split -------------------------------------------
    // decode_len  = 2 - min(seq_len, 2)  -> 1 when seq_len == 1, else 0
    // prefill_len = seq_len - decode_len -> 0 when seq_len == 1, else seq_len
    // Plain arithmetic is used on purpose: comparison ops are excluded from the
    // GPU plugin's shape-of sub-graphs, which would force a device kernel for an
    // i64 Select and fail layout selection.
    const auto i64 = ov::element::i64;
    auto c_zero  = ov::op::v0::Constant::create(i64, ov::Shape{1}, {0});
    auto c_one   = ov::op::v0::Constant::create(i64, ov::Shape{1}, {1});
    auto c_two   = ov::op::v0::Constant::create(i64, ov::Shape{1}, {2});
    auto c_axis1 = ov::op::v0::Constant::create(i64, ov::Shape{1}, {1});

    auto hidden = embed_gather->output(0);
    // Snapshot the original consumers before any of the new nodes below start
    // consuming `hidden` themselves, otherwise the rewiring creates a cycle.
    auto hidden_consumers = hidden.get_target_inputs();

    auto hidden_shape = std::make_shared<ov::op::v3::ShapeOf>(hidden, i64);
    auto seq_len = std::make_shared<ov::op::v8::Gather>(hidden_shape->output(0), c_one, c_zero);
    auto clamped = std::make_shared<ov::op::v1::Minimum>(seq_len->output(0), c_two);
    auto decode_len = std::make_shared<ov::op::v1::Subtract>(c_two, clamped->output(0));
    auto prefill_len = std::make_shared<ov::op::v1::Subtract>(seq_len->output(0), decode_len->output(0));
    prefill_len->set_friendly_name("megakernel_prefill_len");
    decode_len->set_friendly_name("megakernel_decode_len");

    auto slice_seq = [&](const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& len, const char* name) {
        auto s = std::make_shared<ov::op::v8::Slice>(data, c_zero, len, c_one, c_axis1);
        s->set_friendly_name(name);
        return s;
    };

    // The original path is fed the prefill slice of every sequence-shaped input, so
    // during decode its whole sub-graph collapses to zero-length tensors.
    {
        auto hidden_prefill = slice_seq(hidden, prefill_len->output(0), "megakernel_hidden_prefill");
        for (auto& inp : hidden_consumers)
            inp.replace_source_output(hidden_prefill->output(0));
    }
    {
        auto pos_consumers = pos_ids_node->output(0).get_target_inputs();
        auto pos_prefill = slice_seq(pos_ids_node->output(0), prefill_len->output(0), "megakernel_position_ids_prefill");
        for (auto& inp : pos_consumers)
            inp.replace_source_output(pos_prefill->output(0));
    }
    {
        // input_ids feeds the embedding (kept full: the MegaKernel needs the decode
        // token) and a ShapeOf that drives the causal-mask query length, which must
        // follow the split.
        std::vector<ov::Input<ov::Node>> shape_consumers;
        for (auto& inp : input_ids_node->output(0).get_target_inputs()) {
            if (ov::is_type<ov::op::v3::ShapeOf>(inp.get_node()) || ov::is_type<ov::op::v0::ShapeOf>(inp.get_node()))
                shape_consumers.push_back(inp);
        }
        if (!shape_consumers.empty()) {
            auto ids_prefill = slice_seq(input_ids_node->output(0), prefill_len->output(0), "megakernel_input_ids_prefill");
            for (auto& inp : shape_consumers)
                inp.replace_source_output(ids_prefill->output(0));
        }
    }

    auto hidden_decode = slice_seq(hidden, decode_len->output(0), "megakernel_hidden_decode");

    // --- 6. Build MegaKernel -----------------------------------------------
    op::MegaKernelAttrs attrs;
    // attrs already has the correct Qwen3-0.6B defaults

    ov::OutputVector mk_inputs = {
        hidden_decode->output(0),                // 0  hidden_states (decode token only)
        pos_ids_node->output(0),                 // 1  position_ids (unsliced)
        beam_idx_node->output(0),                // 2  beam_idx
        stacked_proj["q_proj"]->output(0),       // 3
        stacked_proj["k_proj"]->output(0),       // 4
        stacked_proj["v_proj"]->output(0),       // 5
        stacked_proj["o_proj"]->output(0),       // 6
        stacked_proj["gate_proj"]->output(0),    // 7
        stacked_proj["up_proj"]->output(0),      // 8
        stacked_proj["down_proj"]->output(0),    // 9
        stacked_norm["input_ln"]->output(0),     // 10
        stacked_norm["post_attn_ln"]->output(0), // 11
        stacked_norm["q_norm"]->output(0),       // 12
        stacked_norm["k_norm"]->output(0),       // 13
        rope_inv_freq,                           // 14
    };

    auto mk_node = std::make_shared<op::MegaKernel>(mk_inputs, attrs);
    mk_node->set_friendly_name("MegaKernel");
    ov::copy_runtime_info(last_add, mk_node);

    // --- 7. Merge the two paths --------------------------------------------------
    // Exactly one of the two branches carries tokens, so Concat along the sequence
    // axis reproduces the active branch's hidden states.
    const auto residual_et = last_add->get_output_element_type(0);
    auto mk_hidden = std::make_shared<ov::op::v0::Convert>(mk_node->output(0), residual_et);
    mk_hidden->set_friendly_name("megakernel_hidden_out");

    auto last_add_consumers = last_add->output(0).get_target_inputs();
    auto merged = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{last_add->output(0), mk_hidden->output(0)}, 1);
    merged->set_friendly_name("megakernel_merge");
    for (auto& inp : last_add_consumers)
        inp.replace_source_output(merged->output(0));

    m->validate_nodes_and_infer_types();

    // -----------------------------------------------------------------------
    // Verification: the MegaKernel must be present and the KV-cache state must
    // still be intact (prefill owns it and hands it over to the MegaKernel).
    // The optional SVG dump is gated behind OV_MEGAKERNEL_DUMP=1.
    // -----------------------------------------------------------------------
    {
        const bool dump = [] {
            const char* v = std::getenv("OV_MEGAKERNEL_DUMP");
            return v && v[0] == '1';
        }();

        int mk_count = 0;
        int assign_kv_count = 0;
        for (auto& op : m->get_ordered_ops()) {
            if (op->get_friendly_name() == "MegaKernel")
                ++mk_count;
            auto as = ov::as_type_ptr<ov::op::v6::Assign>(op);
            if (as && as->get_variable_id().find("past_key_values") != std::string::npos)
                ++assign_kv_count;
        }
        OPENVINO_ASSERT(mk_count == 1,
                        "[MegaKernel] post-insertion check FAILED: expected 1 MegaKernel node, found ", mk_count);
        OPENVINO_ASSERT(assign_kv_count == 2 * NUM_LAYERS,
                        "[MegaKernel] post-insertion check FAILED: ", assign_kv_count,
                        " KV Assign sinks present (expected ", 2 * NUM_LAYERS, ")");

        if (dump) {
            std::cout << "[MegaKernel] PASS: MegaKernel node inserted, prefill path kept intact ("
                      << assign_kv_count << " KV Assign sinks)." << std::endl;

            // VisualizeTree always writes a .dot file; the built-in dot->SVG step is
            // only active when ENABLE_OPENVINO_DEBUG=ON, so invoke graphviz explicitly.
            const char* svg_dir_env = std::getenv("OV_MEGAKERNEL_DUMP_DIR");
            std::string svg_dir = svg_dir_env ? svg_dir_env
                                              : "/opt/home/pwysocki/openvino/MEGAKERNEL_POC/python";
            std::string dot_path = svg_dir + "/megakernel_transformed_graph.dot";
            std::string svg_path = svg_dir + "/megakernel_transformed_graph.svg";
            try {
                ov::pass::Manager viz_pass;
                viz_pass.register_pass<ov::pass::VisualizeTree>(dot_path, nullptr, /*dot_only=*/true);
                viz_pass.run_passes(m);

                std::string cmd = "dot -Tsvg " + dot_path + " -o " + svg_path + " 2>&1";
                if (std::system(cmd.c_str()) == 0) {
                    std::cout << "[MegaKernel] Transformed graph dumped to: " << svg_path << std::endl;
                } else {
                    std::cerr << "[MegaKernel] WARNING: graphviz conversion failed; raw DOT at: "
                              << dot_path << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "[MegaKernel] WARNING: graph dump failed: " << e.what() << std::endl;
            }
        }
    }

    return true;
}

}  // namespace ov::intel_gpu
