#pragma once

// Helpers shared by the per-model megakernel transformations
// (see iMegakernelTransformation.h). Everything here is architecture-agnostic:
// anything that depends on a particular model's layer count, weight names or
// megakernel input order belongs in that model's transformation instead.

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace mk::util {

// Return the node with the given friendly name or nullptr.
inline std::shared_ptr<ov::Node> find_node(const ov::NodeVector& ops, const std::string& name) {
    for (auto& op : ops) {
        if (op->get_friendly_name() == name)
            return op;
    }
    return nullptr;
}

// Walk through Convert wrappers to the underlying Constant and return its data
// as a flat float16 (f16) buffer re-packaged into a new Constant of the given
// shape.  If the node is already a Constant of dtype f16 it is returned as-is.
inline std::shared_ptr<ov::op::v0::Constant> get_f16_constant(std::shared_ptr<ov::Node> node) {
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
    if (c->get_element_type() == ov::element::f16)
        return c;
    auto data_f32 = c->cast_vector<ov::float16>();
    return ov::op::v0::Constant::create(ov::element::f16, c->get_shape(), data_f32);
}

// Stack a vector of per-layer Constants along a new leading dimension.
// Each constant must have the same shape.  Returns new stacked Constant.
inline std::shared_ptr<ov::op::v0::Constant> stack_constants(
        const std::vector<std::shared_ptr<ov::op::v0::Constant>>& per_layer,
        const std::string& debug_name) {

    OPENVINO_ASSERT(!per_layer.empty());
    const ov::Shape elem_shape = per_layer[0]->get_shape();
    const ov::element::Type dtype = per_layer[0]->get_element_type();

    size_t elem_size = dtype.bitwidth() / 8;
    size_t layer_bytes = ov::shape_size(elem_shape) * elem_size;
    std::vector<uint8_t> buf(per_layer.size() * layer_bytes);

    for (size_t i = 0; i < per_layer.size(); ++i) {
        OPENVINO_ASSERT(per_layer[i]->get_element_type() == dtype,
                        "[MegaKernel] dtype mismatch while stacking ", debug_name);
        OPENVINO_ASSERT(per_layer[i]->get_shape() == elem_shape,
                        "[MegaKernel] shape mismatch while stacking ", debug_name);
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
inline std::shared_ptr<ov::op::v0::Constant> squeeze_leading_ones(
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
// The constant's friendly name is "<prefix><layer>.<weight_name_suffix>"
inline std::shared_ptr<ov::op::v0::Constant> get_proj_weight(
        const ov::NodeVector& ops, int layer, const std::string& weight_name_suffix,
        const std::string& prefix = "self.model.layers.") {
    const std::string expected = prefix + std::to_string(layer) + "." + weight_name_suffix;
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
inline std::shared_ptr<ov::op::v0::Constant> get_norm_weight(
        const ov::NodeVector& ops, int layer, const std::string& norm_fragment) {
    const std::string target_frag = "layers." + std::to_string(layer) + "." + norm_fragment;
    for (auto& op : ops) {
        const std::string& n = op->get_friendly_name();
        if (n.find(target_frag) == std::string::npos) continue;
        if (n.find("Multiply_1") == std::string::npos) continue;
        // Try both input ports - weight may be at port 0 or 1 depending on export order.
        for (size_t port = 0; port < op->get_input_size(); ++port) {
            auto candidate = op->get_input_node_shared_ptr(port);
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

// Number of "past_key_values" ReadValue nodes in the graph. Transformations use
// this as a cheap architecture fingerprint (2 * num_layers for a plain KV cache).
inline int count_kv_read_values(const ov::NodeVector& ops) {
    int n = 0;
    for (auto& op : ops) {
        auto rv = ov::as_type_ptr<ov::op::v6::ReadValue>(op);
        if (rv && rv->get_variable_id().find("past_key_values") != std::string::npos)
            ++n;
    }
    return n;
}

// Parse the layer index out of a "past_key_values.<l>.<suffix>" variable id.
// Returns false if the id does not have that shape.
inline bool parse_kv_variable_id(const std::string& vid, int& layer, bool& is_key) {
    if (vid.find("past_key_values") == std::string::npos) return false;
    auto dot1 = vid.find('.');
    auto dot2 = vid.find('.', dot1 + 1);
    if (dot1 == std::string::npos || dot2 == std::string::npos) return false;
    layer = std::stoi(vid.substr(dot1 + 1, dot2 - dot1 - 1));
    // The suffix after the layer number disambiguates key from value; a naive
    // search for "key"/"value" would match "past_key_values" in every id.
    is_key = vid.substr(dot2 + 1).rfind("key", 0) == 0;
    return true;
}

// Collect layer-sorted ReadValue ops for the key and value caches.
inline void collect_sorted_rv(const ov::NodeVector& ops, int num_layers,
                              std::vector<std::shared_ptr<ov::op::v6::ReadValue>>& key_rvs,
                              std::vector<std::shared_ptr<ov::op::v6::ReadValue>>& val_rvs) {
    std::map<int, std::shared_ptr<ov::op::v6::ReadValue>> key_map, val_map;
    for (auto& op : ops) {
        auto rv = ov::as_type_ptr<ov::op::v6::ReadValue>(op);
        if (!rv) continue;
        int layer = 0;
        bool is_key = false;
        if (!parse_kv_variable_id(rv->get_variable_id(), layer, is_key)) continue;
        (is_key ? key_map : val_map)[layer] = rv;
    }
    key_rvs.clear(); val_rvs.clear();
    for (int l = 0; l < num_layers; ++l) {
        OPENVINO_ASSERT(key_map.count(l), "[MegaKernel] missing key ReadValue for layer ", l);
        OPENVINO_ASSERT(val_map.count(l), "[MegaKernel] missing val ReadValue for layer ", l);
        key_rvs.push_back(key_map.at(l));
        val_rvs.push_back(val_map.at(l));
    }
}

// Collect layer-sorted Assign ops for the key and value caches.
inline void collect_sorted_assign(const ov::NodeVector& ops, int num_layers,
                                  std::vector<std::shared_ptr<ov::op::v6::Assign>>& key_as,
                                  std::vector<std::shared_ptr<ov::op::v6::Assign>>& val_as) {
    std::map<int, std::shared_ptr<ov::op::v6::Assign>> key_map, val_map;
    for (auto& op : ops) {
        auto as = ov::as_type_ptr<ov::op::v6::Assign>(op);
        if (!as) continue;
        int layer = 0;
        bool is_key = false;
        if (!parse_kv_variable_id(as->get_variable_id(), layer, is_key)) continue;
        (is_key ? key_map : val_map)[layer] = as;
    }
    key_as.clear(); val_as.clear();
    for (int l = 0; l < num_layers; ++l) {
        OPENVINO_ASSERT(key_map.count(l), "[MegaKernel] missing key Assign for layer ", l);
        OPENVINO_ASSERT(val_map.count(l), "[MegaKernel] missing val Assign for layer ", l);
        key_as.push_back(key_map.at(l));
        val_as.push_back(val_map.at(l));
    }
}

// Build [num_layers, B, num_kv_heads, S_past, head_dim] by stacking the
// per-layer ReadValue outputs via Unsqueeze(axis=0) + Concat(axis=0).
inline ov::Output<ov::Node> build_stacked_kv(
        const std::vector<std::shared_ptr<ov::op::v6::ReadValue>>& rvs) {
    auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    ov::OutputVector unsqueezed;
    unsqueezed.reserve(rvs.size());
    for (auto& rv : rvs) {
        auto us = std::make_shared<ov::op::v0::Unsqueeze>(rv->output(0), axis_const);
        unsqueezed.push_back(us->output(0));
    }
    auto cat = std::make_shared<ov::op::v0::Concat>(unsqueezed, 0);
    return cat->output(0);
}

// Locate the rotary-embedding inverse-frequency Constant by its element count
// and by having a consumer whose name mentions "rotary".
inline ov::Output<ov::Node> find_rope_inv_freq(const ov::NodeVector& ops, size_t expected_elements) {
    for (auto& op : ops) {
        auto c = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!c || ov::shape_size(c->get_shape()) != expected_elements) continue;
        for (auto& out : c->outputs()) {
            for (auto& inp : out.get_target_inputs()) {
                if (inp.get_node()->get_friendly_name().find("rotary") != std::string::npos)
                    return c->output(0);
            }
        }
    }
    OPENVINO_THROW("[MegaKernel] rope inv_freq not found");
}

// Post-insertion sanity checks plus the optional graph dump.
// Enabled with OV_MEGAKERNEL_DUMP=1, output directory via OV_MEGAKERNEL_DUMP_DIR.
inline void verify_and_dump(const std::shared_ptr<ov::Model>& m, int expected_removed_assigns,
                            int expected_mk_nodes = 1, const std::string& mk_name_prefix = "MegaKernel") {
    const bool dump = [] {
        const char* v = std::getenv("OV_MEGAKERNEL_DUMP");
        return v && v[0] == '1';
    }();

    int mk_count = 0;
    int assign_kv_count = 0;
    for (auto& op : m->get_ordered_ops()) {
        if (op->get_friendly_name().rfind(mk_name_prefix, 0) == 0)
            ++mk_count;
        auto as = ov::as_type_ptr<ov::op::v6::Assign>(op);
        if (as && as->get_variable_id().find("past_key_values") != std::string::npos)
            ++assign_kv_count;
    }
    OPENVINO_ASSERT(mk_count == expected_mk_nodes,
                    "[MegaKernel] post-insertion check FAILED: expected ", expected_mk_nodes,
                    " MegaKernel node(s), found ", mk_count);
    OPENVINO_ASSERT(assign_kv_count == expected_removed_assigns,
                    "[MegaKernel] post-insertion check FAILED: ", assign_kv_count,
                    " KV Assign sinks present (expected ", expected_removed_assigns, ")");

    if (!dump)
        return;

    std::cout << "[MegaKernel] PASS: MegaKernel node successfully inserted (" << mk_count << " node)." << std::endl;
    std::cout << "[MegaKernel] PASS: all " << expected_removed_assigns
              << " KV-cache Assign sinks removed." << std::endl;

    // VisualizeTree always writes a .dot file; the built-in dot->SVG step is only
    // active when ENABLE_OPENVINO_DEBUG=ON, so invoke graphviz explicitly.
    const char* svg_dir_env = std::getenv("OV_MEGAKERNEL_DUMP_DIR");
    std::string svg_dir = svg_dir_env ? svg_dir_env : ".";
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

}  // namespace mk::util
