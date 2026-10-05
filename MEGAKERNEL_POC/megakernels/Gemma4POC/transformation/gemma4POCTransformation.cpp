// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MegaKernel graph transformation for Gemma-4-26B-A4B-it.
//
// WHY THESE TARGETS
// -----------------
// A decode step on B60 costs 11.8 ms wall for only 8.5 ms of summed device time
// spread over 1002 kernels. The missing 3.3 ms is submission, not compute: a
// perf profile of the decode loop alone puts 63% of host CPU in the NEO driver
// (half of that spinning in clFinish) and only 18% in the GPU plugin. The host
// cannot feed ~1000 enqueues faster than the GPU drains the small ones, so wall
// time tracks the PRIMITIVE COUNT, not FLOPs or bytes. Removing 109 primitives
// was measured to be worth 0.5 ms, i.e. ~4.6 us of wall time each.
//
// Consequences, all measured rather than assumed:
//   * lm_head (1.9 ms, one kernel) runs at ~85% of the memory roofline, and a
//     hand-written replacement came out slower - see OV_MEGAKERNEL_LMHEAD below.
//   * MOECompressed is 0.3 ms over 30 kernels; already fused upstream.
//   * The int4 projections are 2-3x off roofline but are only 156 kernels.
//   * ~530 of the 1002 kernels are norms and element-wise glue moving a few KB
//     each. That is what this pass attacks.
// Things that sound promising and are not: an in-order command queue
// (OV_GPU_QUEUE_TYPE=in-order) is a no-op, and reshaping the model to a static
// one-token shape buys only 3%.
//
// This pass runs after the plugin's RMS and weight-compression fusions, so what
// is left here is what those passes could not match.

#include "../../iMegakernelTransformation.h"
#include "../../megakernelTransformationUtils.h"
#include "../gemma4POCParams.h"
#include "../gemma4Repack.h"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/megakernel.hpp"
#include "intel_gpu/op/moe_router_fused.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace mk {
namespace {

constexpr char LM_HEAD[] = "__module.lm_head/ov_ext::linear/MatMul";

bool is_dbg() {
    const char* v = std::getenv("OV_MEGAKERNEL_DUMP");
    return v && v[0] == '1';
}

using RMSOp = ov::op::internal::RMS;
using RoPEOp = ov::op::internal::RoPE;

// Walk back through Convert nodes to the Constant feeding them, if any.
std::shared_ptr<v0::Constant> as_constant(const ov::Output<ov::Node>& out) {
    auto n = out.get_node_shared_ptr();
    for (int d = 0; d < 4 && n && !ov::is_type<v0::Constant>(n); ++d) {
        if (!ov::is_type<v0::Convert>(n))
            return nullptr;
        n = n->get_input_node_shared_ptr(0);
    }
    return ov::as_type_ptr<v0::Constant>(n);
}

// The gamma operand of an RMS, or nullptr for the gamma-less form.
std::shared_ptr<v0::Constant> rms_gamma(const std::shared_ptr<RMSOp>& rms) {
    return rms->get_input_size() < 2 ? nullptr : as_constant(rms->input_value(1));
}

// ---------------------------------------------------------------------------
// Pass 1: fuse the gamma-less RMSNorm that ov::pass::RMSFusion leaves behind
// ---------------------------------------------------------------------------
// Gemma-4's v_norm is x * (mean(x^2) + eps)^-0.5 with no learned scale. Upstream
// RMSFusion only fires when the rsqrt is followed by a multiply (by gamma, or by
// a dynamic scale); here it is followed by a Transpose, so the chain stays
// decomposed into Power + ReduceMean + Multiply - three dispatches per layer for
// 16 KB of traffic. cldnn's rms primitive accepts the one-input form, so the
// rewrite is exact; gemma4_rms_check.py confirms that form is in fact slightly
// MORE accurate than the decomposition it replaces.
// Fires 30x (25 v_norm plus the k_norm of the 5 full-attention layers, where
// attention_k_eq_v makes k and v share one tensor). Worth ~3.5%.
}  // namespace

// Not in the anonymous namespace: OPENVINO_MATCHER_PASS_RTTI attaches a
// visibility attribute, which is ignored (and -Werror) on internal-linkage types.
class FuseGammalessRMS : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Gemma4FuseGammalessRMS");

    explicit FuseGammalessRMS(int& hits) {
        using namespace ov::pass::pattern;

        auto x = any_input();
        auto square = wrap_type<v1::Power>({x, optional<v0::Convert>(wrap_type<v0::Constant>(value_matches("2")))});
        auto axes = wrap_type<v0::Constant>();
        auto mean = wrap_type<v1::ReduceMean>({square, axes});
        auto eps = wrap_type<v0::Constant>();
        auto add_eps = wrap_type<v1::Add>({mean, optional<v0::Convert>(eps)});
        auto rsqrt =
            wrap_type<v1::Power>({add_eps, optional<v0::Convert>(wrap_type<v0::Constant>(value_matches("-0.5")))});
        auto mul = wrap_type<v1::Multiply>({x, rsqrt});

        ov::matcher_pass_callback cb = [=, &hits](Matcher& m) {
            const auto& pm = m.get_pattern_value_map();
            auto root = m.get_match_root();

            auto eps_c = ov::as_type_ptr<v0::Constant>(pm.at(eps).get_node_shared_ptr());
            auto axes_c = ov::as_type_ptr<v0::Constant>(pm.at(axes).get_node_shared_ptr());
            if (!eps_c || !axes_c || ov::shape_size(eps_c->get_shape()) != 1 ||
                ov::shape_size(axes_c->get_shape()) != 1)
                return false;

            // RMS normalises over the trailing axis only.
            const auto mean_node = pm.at(mean).get_node_shared_ptr();
            const auto rank = static_cast<int64_t>(mean_node->get_input_partial_shape(0).size());
            const int64_t axis = axes_c->cast_vector<int64_t>()[0];
            if (axis != -1 && axis != rank - 1)
                return false;

            auto rms =
                std::make_shared<RMSOp>(pm.at(x), eps_c->cast_vector<float>()[0], root->get_output_element_type(0));
            rms->set_friendly_name(root->get_friendly_name());
            ov::copy_runtime_info(m.get_matched_nodes(), rms);
            ov::replace_node(root, rms);
            ++hits;
            if (is_dbg())
                std::cerr << "[Gemma4]   gammaless " << root->get_friendly_name() << " shape "
                          << root->get_output_partial_shape(0) << " type " << root->get_output_element_type(0)
                          << std::endl;
            return true;
        };

        register_matcher(std::make_shared<Matcher>(mul, "Gemma4FuseGammalessRMS"), cb);
    }
};

namespace {

// ---------------------------------------------------------------------------
// Pass 2: fold a sibling RMS's gamma into the weights of the matmul it feeds
// ---------------------------------------------------------------------------
// Gemma-4 normalises the post-attention residual twice with the same statistics
// but different scales: once for the feed-forward branch and once for the MoE
// router. After RMSFusion that is two independent RMS primitives over the same
// input. router.proj is a plain matmul with a dense constant weight, so
// gamma_router can be pushed into that weight as gamma_router / gamma_ff, which
// lets the router reuse the feed-forward RMS and removes one dispatch per layer.
// Fires on 19 of the 30 layers; the rest have a gamma_ff channel small enough
// that the ratio guard below rejects it. Logits stay at cosine 0.999997.
int fold_sibling_rms_gamma(const std::shared_ptr<ov::Model>& m, bool dbg) {
    int folded = 0;
    const std::string trace_layer = "layers.7.";
    auto trace = [&](const std::shared_ptr<RMSOp>& r, const char* why) {
        if (dbg && r->get_friendly_name().find(trace_layer) != std::string::npos)
            std::cerr << "[Gemma4]   fold-skip " << r->get_friendly_name() << ": " << why << std::endl;
    };

    for (auto& op : m->get_ordered_ops()) {
        auto rms_b = ov::as_type_ptr<RMSOp>(op);
        if (!rms_b || !rms_b->get_elementwise_affine())
            continue;
        auto gamma_b = rms_gamma(rms_b);
        if (!gamma_b)
            continue;

        // Exactly one consumer, and it must expose a dense constant weight whose
        // reduction axis matches gamma.
        auto consumers = rms_b->output(0).get_target_inputs();
        if (consumers.size() != 1 || consumers.begin()->get_index() != 0) {
            trace(rms_b, "not a single port-0 consumer");
            continue;
        }
        ov::Node* mm = consumers.begin()->get_node();
        if (mm->get_input_size() < 2) {
            trace(rms_b, "consumer has no weight input");
            continue;
        }

        const size_t k = ov::shape_size(gamma_b->get_shape());
        auto w = as_constant(mm->input_value(1));
        // Quantised weights carry their own scales and cannot absorb a per-input
        // channel factor, so only dense float weights qualify.
        if (!w || !w->get_element_type().is_real() || w->get_shape().size() != 2 || w->get_shape()[1] != k) {
            trace(rms_b, w ? "weight is not a dense float [N,K] constant" : "weight is not a constant");
            continue;
        }

        // A sibling RMS over the same activation with a different scale.
        std::shared_ptr<RMSOp> rms_a;
        for (const auto& t : rms_b->input_value(0).get_target_inputs()) {
            auto cand = ov::as_type_ptr<RMSOp>(t.get_node()->shared_from_this());
            if (!cand || cand == rms_b || t.get_index() != 0 || !cand->get_elementwise_affine())
                continue;
            if (cand->get_epsilon() != rms_b->get_epsilon())
                continue;
            auto gamma_a = rms_gamma(cand);
            if (gamma_a && ov::shape_size(gamma_a->get_shape()) == k) {
                rms_a = cand;
                break;
            }
        }
        if (!rms_a) {
            trace(rms_b, "no sibling RMS over the same activation");
            continue;
        }
        const auto ga = rms_gamma(rms_a)->cast_vector<float>();
        const auto gb = gamma_b->cast_vector<float>();

        // gamma_ff has entries very close to zero, so a blanket magnitude
        // threshold rejects every layer. What actually matters is that the
        // rescaled weight stays representable in the weight's own dtype.
        const float limit =
            w->get_element_type() == ov::element::f16 ? 65504.0f : std::numeric_limits<float>::max();
        auto wv = w->cast_vector<float>();
        const size_t rows = w->get_shape()[0];
        bool safe = true;
        for (size_t col = 0; col < k && safe; ++col) {
            const float ratio = gb[col] / ga[col];
            // A huge ratio means the shared RMS emits a near-denormal f16 for
            // that channel and the weight then amplifies its rounding error.
            if (!std::isfinite(ratio) || std::abs(ratio) > 1e3f) {
                safe = false;
                break;
            }
            for (size_t row = 0; row < rows; ++row) {
                float v = wv[row * k + col] * ratio;
                if (std::abs(v) > 0.5f * limit) {
                    safe = false;
                    break;
                }
                wv[row * k + col] = v;
            }
        }
        if (!safe) {
            trace(rms_b, "rescaled weight would not be representable");
            continue;
        }

        auto w_scaled = std::make_shared<v0::Constant>(w->get_element_type(), w->get_shape(), wv);
        w_scaled->set_friendly_name(w->get_friendly_name() + "_gamma_folded");
        // input_value(1) may be a Convert wrapper; re-point whichever node reads
        // the original constant.
        auto direct = mm->input_value(1).get_node_shared_ptr();
        if (direct == w)
            mm->input(1).replace_source_output(w_scaled->output(0));
        else
            direct->input(0).replace_source_output(w_scaled->output(0));

        ov::replace_node(rms_b, rms_a);
        ++folded;
        if (dbg)
            std::cerr << "[Gemma4]   fold " << rms_b->get_friendly_name() << " -> "
                      << rms_a->get_friendly_name() << " via " << mm->get_friendly_name() << std::endl;
    }

    if (dbg)
        std::cerr << "[Gemma4] sibling-RMS gammas folded into matmul weights: " << folded << std::endl;
    return folded;
}

// ---------------------------------------------------------------------------
// Pass 3: absorb the post-RoPE Transpose into the RoPE primitive
// ---------------------------------------------------------------------------
// Attention needs q as [B, H, S, D] but RoPE produces [B, S, H, D], so the graph
// carries a separate Transpose(0,2,1,3) per layer. cldnn's rope primitive can
// emit the transposed layout itself via Config::output_trans0213, which upstream
// TransposeFusion does not exploit here. The rotate-half GPU kernel did not
// actually implement that flag (it only changed the declared output shape, so
// enabling it silently corrupted attention); it is implemented now in
// impls/ocl_v2/rope_opt.{cl,cpp}. Bit-exact, and removes 30 live Transposes.
// Not applicable to interleaved RoPE, which the GPU op rejects in combination.
int fold_transpose_into_rope(const std::shared_ptr<ov::Model>& m, bool dbg) {
    int folded = 0;

    for (auto& op : m->get_ordered_ops()) {
        auto tr = ov::as_type_ptr<v1::Transpose>(op);
        if (!tr)
            continue;
        auto order = ov::as_type_ptr<v0::Constant>(tr->get_input_node_shared_ptr(1));
        if (!order || order->cast_vector<int64_t>() != std::vector<int64_t>{0, 2, 1, 3})
            continue;

        auto rope = ov::as_type_ptr<RoPEOp>(tr->get_input_node_shared_ptr(0));
        if (!rope || rope->output(0).get_target_inputs().size() != 1)
            continue;

        auto cfg = rope->get_config();
        if (cfg.output_trans0213 || cfg.is_interleaved)
            continue;

        cfg.output_trans0213 = true;
        auto fused = std::make_shared<RoPEOp>(rope->input_values(), cfg);
        fused->set_friendly_name(rope->get_friendly_name());
        ov::copy_runtime_info({rope, tr}, fused);
        ov::replace_node(tr, fused);
        ++folded;
    }

    if (dbg)
        std::cerr << "[Gemma4] transposes absorbed into RoPE: " << folded << std::endl;
    return folded;
}


// ---------------------------------------------------------------------------
// Opt-in: replace lm_head with the hand-written INT8 GEMV
// ---------------------------------------------------------------------------
// MEASURED: this replacement is a REGRESSION and is therefore opt-in.
//   baseline lm_head (OpenVINO)  1.92 ms   = ~85% of the memory roofline
//   this kernel, 3 variants      ~2.5 ms
//   end-to-end decode            11.65 -> 12.08 ms/tok (-3.8%)
// Accuracy is fine (cosine 0.999999, argmax match, top-10 10/10), so the kernel
// is correct - it simply loses to the stock int8 path, which almost certainly
// uses dp4a-style integer dot products rather than float MACs. Kept because it
// is a working, validated INT8 GEMV to build on; enable with
// OV_MEGAKERNEL_LMHEAD=1.
bool replace_lm_head(const std::shared_ptr<ov::Model>& m, bool dbg) {
    namespace mku = mk::util;
    using ov::intel_gpu::op::MegaKernel;
    using ov::intel_gpu::op::MegaKernelAttrs;

    auto lm = mku::find_node(m->get_ordered_ops(), LM_HEAD);
    if (!lm) {
        if (dbg)
            std::cerr << "[Gemma4] lm_head not found; skipping" << std::endl;
        return false;
    }

    // Port 0 is the activation and port 1 the packed weights; the decompression
    // operands are identified by element type rather than by a fixed index.
    ov::Output<ov::Node> act = lm->input_value(0);
    ov::Output<ov::Node> wq = lm->input_value(1);
    ov::Output<ov::Node> wscale, wzp;

    for (size_t i = 2; i < lm->get_input_size(); ++i) {
        auto src = lm->input_value(i);
        const auto et = src.get_element_type();
        if (et == ov::element::f16 || et == ov::element::f32 || et == ov::element::bf16) {
            if (!wscale.get_node())
                wscale = src;
        } else if (et == ov::element::u4 || et == ov::element::u8 || et == ov::element::i4) {
            if (!wzp.get_node())
                wzp = src;
        }
    }

    if (wq.get_element_type() != ov::element::u8 || !wscale.get_node() || !wzp.get_node()) {
        if (dbg)
            std::cerr << "[Gemma4] lm_head is not u8+scale+zp; skipping" << std::endl;
        return false;
    }

    const auto wshape = wq.get_partial_shape();
    if (wshape.rank().is_dynamic() || wshape.rank().get_length() != 2 || wshape.is_dynamic()) {
        if (dbg)
            std::cerr << "[Gemma4] lm_head weight shape not static 2D; skipping" << std::endl;
        return false;
    }

    const int64_t N = wshape[0].get_length();
    const int64_t K = wshape[1].get_length();

    // Only the per-row (single group) form is handled; anything else falls back.
    const auto sshape = wscale.get_partial_shape();
    if (sshape.rank().is_dynamic() || ov::shape_size(sshape.to_shape()) != static_cast<size_t>(N)) {
        if (dbg)
            std::cerr << "[Gemma4] lm_head scale is not per-row; skipping" << std::endl;
        return false;
    }

    MegaKernelAttrs attrs;
    attrs.hidden_size = N;        // output width
    attrs.intermediate_size = K;  // reduction depth

    auto mk_node = std::make_shared<MegaKernel>(ov::OutputVector{act, wq, wzp, wscale}, attrs);
    mk_node->set_friendly_name("MegaKernelLMHead");
    ov::copy_runtime_info(lm, mk_node);

    // The op emits f32; lm_head's consumers (logit soft-capping) are f16.
    auto mk_cvt = std::make_shared<v0::Convert>(mk_node->output(0), lm->output(0).get_element_type());
    mk_cvt->set_friendly_name("lm_head_mk_cvt");

    for (auto& c : lm->output(0).get_target_inputs())
        c.replace_source_output(mk_cvt->output(0));

    if (dbg)
        std::cerr << "[Gemma4] lm_head replaced: N=" << N << " K=" << K << std::endl;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Pass 4: replace each layer's whole feed-forward block with one MegaKernel
// ---------------------------------------------------------------------------
// MEASURED (cltrace, one decode step): from o_proj's output to the layer_scalar
// multiply a layer runs ~17 kernels for ~192 us, of which the MoE gate_up/down
// GEMVs alone are 102 us at ~230 GB/s (half of B60's 456 GB/s) and the 9 RMS /
// router / reduce kernels another ~45 us. The MegaKernel runs the same math as
// 4 launches: RMS statistics are recomputed per work-group instead of being
// separate dispatches, the dense MLP and router share one launch, the MoE gate/up
// shares one with the dense down projection, and MoE down folds the top-k
// weighted sum into the GEMV. Intermediates stay f32.
namespace {

using FCC = ov::intel_gpu::op::FullyConnectedCompressed;
using FC = ov::intel_gpu::op::FullyConnected;
using Router = ov::intel_gpu::op::MoERouterFused;
using MOEC = ov::op::internal::MOECompressed;

ov::Output<ov::Node> skip_reshape(ov::Output<ov::Node> v) {
    while (ov::is_type<v1::Reshape>(v.get_node()))
        v = v.get_node()->input_value(0);
    return v;
}

std::shared_ptr<RMSOp> as_rms(const ov::Output<ov::Node>& v) {
    auto r = ov::as_type_ptr<RMSOp>(v.get_node_shared_ptr());
    if (!r || std::abs(r->get_epsilon() - 1e-6) > 1e-9 || !rms_gamma(r))
        return nullptr;
    return r;
}

ov::Output<ov::Node> f16_gamma(const std::shared_ptr<RMSOp>& r) {
    auto g = rms_gamma(r);
    if (ov::shape_size(g->get_shape()) != static_cast<size_t>(kHidden))
        return {};
    return mk::util::get_f16_constant(g)->output(0);
}

// An int4 FullyConnectedCompressed with a [N,K] weight, [N,K/64] f16 scale and
// u8 zero point; returns {W, scale, zp} or an empty vector.
ov::OutputVector int4_fc(const std::shared_ptr<ov::Node>& n, size_t N, size_t K) {
    auto fc = ov::as_type_ptr<FCC>(n);
    if (!fc || fc->get_input_size() != 5 || fc->get_output_size() != 1)
        return {};
    if (std::string(fc->get_input_node_ptr(2)->get_type_name()) != "Placeholder")
        return {};
    auto w = fc->input_value(1), s = fc->input_value(3), z = fc->input_value(4);
    for (const auto& c : {w, s, z})
        if (!ov::is_type<v0::Constant>(c.get_node()))
            return {};
    if (w.get_element_type() != ov::element::u4 || w.get_shape() != ov::Shape{N, K} ||
        s.get_element_type() != ov::element::f16 || s.get_shape() != ov::Shape{N, K / kQuantGroup} ||
        z.get_element_type() != ov::element::u8 || z.get_shape() != ov::Shape{N, K / kQuantGroup})
        return {};
    return {w, s, z};
}

// Rewrites an int4 matrix (rows x K, group-64 scales and zero points) into the
// sub-group record layout of gemma4Repack.h. The original constants are only
// referenced by the replaced subgraph, so the device holds a single copy.
ov::Output<ov::Node> repack(const ov::Output<ov::Node>& w, const ov::Output<ov::Node>& s,
                            const ov::Output<ov::Node>& z, size_t rows, size_t K) {
    auto wc = ov::as_type_ptr<v0::Constant>(w.get_node_shared_ptr());
    auto sc = ov::as_type_ptr<v0::Constant>(s.get_node_shared_ptr());
    auto zc = ov::as_type_ptr<v0::Constant>(z.get_node_shared_ptr());
    const bool zp_u4 = zc->get_element_type() == ov::element::u4;
    auto out = std::make_shared<v0::Constant>(ov::element::u8, ov::Shape{repacked_bytes(rows, K)});
    auto* dst = static_cast<uint8_t*>(const_cast<void*>(out->get_data_ptr()));
    const auto* wp = static_cast<const uint8_t*>(wc->get_data_ptr());
    const auto* sp = static_cast<const uint16_t*>(sc->get_data_ptr());
    const auto* zp = static_cast<const uint8_t*>(zc->get_data_ptr());
    const size_t blocks = rows / kRepackRows;
    const size_t chunk = 64;
    ov::parallel_for((blocks + chunk - 1) / chunk, [&](size_t c) {
        repack_row_blocks(wp, sp, zp, zp_u4, K, c * chunk, std::min(blocks, (c + 1) * chunk), dst);
    });
    return out->output(0);
}

// Replaces the block that MOECompressed `moe` belongs to with a MegaKernel;
// returns false with the graph untouched if anything does not match.
bool replace_ffn_block(const std::shared_ptr<MOEC>& moe, bool dbg) {
    using ov::intel_gpu::op::MegaKernel;
    using ov::intel_gpu::op::MegaKernelAttrs;
    auto reject = [&](const char* why) {
        if (dbg)
            std::cerr << "[Gemma4]   ffn-skip " << moe->get_friendly_name() << ": " << why << std::endl;
        return false;
    };

    const auto& cfg = moe->get_config();
    if (cfg.hidden_size != kHidden || cfg.inter_size != kExpertInter || cfg.num_expert != kExperts ||
        cfg.top_k != kTopK || cfg.group_size != kQuantGroup || !cfg.has_zp || cfg.num_shared_expert != 0 ||
        cfg.expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU ||
        cfg.activation_type != ov::op::internal::MOE::Activation_type::GEGLU_TANH ||
        moe->get_input_size() != 12)
        return reject("MoE config");
    const ov::Shape gu_w{kExperts, kExpertInter, kHidden}, gu_s{kExperts, kExpertInter, kHidden / kQuantGroup};
    const ov::Shape dn_w{kExperts, kHidden, kExpertInter}, dn_s{kExperts, kHidden, kExpertInter / kQuantGroup};
    const ov::Shape want[9] = {gu_w, gu_s, gu_s, gu_w, gu_s, gu_s, dn_w, dn_s, dn_s};
    const ov::element::Type want_t[3] = {ov::element::u4, ov::element::f16, ov::element::u4};
    for (size_t i = 0; i < 9; ++i) {
        auto v = moe->input_value(3 + i);
        if (!ov::is_type<v0::Constant>(v.get_node()) || v.get_shape() != want[i] ||
            v.get_element_type() != want_t[i % 3])
            return reject("MoE weight layout");
    }

    // Router: RMS -> FullyConnected(f16 [E,H]) -> MoERouterFused(softmax, top-8)
    auto router = ov::as_type_ptr<Router>(moe->get_input_node_shared_ptr(1));
    if (!router || moe->get_input_node_ptr(2) != router.get() ||
        router->get_config().routing_type != Router::RoutingType::SOFTMAX || router->get_config().top_k != kTopK)
        return reject("router");
    auto rfc = ov::as_type_ptr<FC>(router->get_input_node_shared_ptr(0));
    if (!rfc || ov::is_type<FCC>(rfc) || !ov::is_type<v0::Constant>(rfc->get_input_node_ptr(1)) ||
        rfc->get_input_element_type(1) != ov::element::f16 ||
        rfc->get_input_shape(1) != ov::Shape{kExperts, kHidden} ||
        std::string(rfc->get_input_node_ptr(2)->get_type_name()) != "Placeholder")
        return reject("router proj");
    auto rms_router = as_rms(rfc->input_value(0));
    auto rms_pf2 = as_rms(moe->input_value(0));
    if (!rms_router || !rms_pf2)
        return reject("router/pre_ff_2 RMS");

    // h1 = h + RMS_pa(o)
    auto h1 = skip_reshape(rms_pf2->input_value(0));
    auto add1 = ov::as_type_ptr<v1::Add>(h1.get_node_shared_ptr());
    if (!add1 || skip_reshape(rms_router->input_value(0)) != h1)
        return reject("h1");
    std::shared_ptr<RMSOp> rms_pa;
    ov::Output<ov::Node> h;
    for (size_t i = 0; i < 2 && !rms_pa; ++i) {
        rms_pa = as_rms(add1->input_value(i));
        h = add1->input_value(1 - i);
    }
    if (!rms_pa)
        return reject("post_attention RMS");
    // o = o_proj(attn), int4 [H, KA] with KA = heads * head_dim of this layer.
    auto oproj = rms_pa->get_input_node_shared_ptr(0);
    const size_t ka = ov::is_type<FCC>(oproj) && oproj->get_input_partial_shape(1).is_static() &&
                              oproj->get_input_shape(1).size() == 2
                          ? oproj->get_input_shape(1)[1]
                          : 0;
    auto oproj_w = ka ? int4_fc(oproj, kHidden, ka) : ov::OutputVector{};
    if (oproj_w.empty() || ka % (kQuantGroup * 8) || ka > 8192 || oproj->output(0).get_target_inputs().size() != 1)
        return reject("o_proj");
    auto attn = oproj->input_value(0);

    // MoE output -> RMS_pf2post; dense down -> RMS_pf1; their sum -> RMS_post
    if (moe->output(0).get_target_inputs().size() != 1)
        return reject("MoE consumers");
    auto after_moe = moe->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
    if (ov::is_type<v1::Reshape>(after_moe)) {
        if (after_moe->output(0).get_target_inputs().size() != 1)
            return reject("MoE reshape consumers");
        after_moe = after_moe->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
    }
    auto rms_pf2post = as_rms(after_moe->output(0));
    if (!rms_pf2post || rms_pf2post->output(0).get_target_inputs().size() != 1)
        return reject("post_ff_2 RMS");
    auto add_s = ov::as_type_ptr<v1::Add>(
        rms_pf2post->output(0).get_target_inputs().begin()->get_node()->shared_from_this());
    if (!add_s)
        return reject("moe+mlp add");
    auto rms_pf1 = as_rms(add_s->input_value(add_s->get_input_node_ptr(0) == rms_pf2post.get() ? 1 : 0));
    if (!rms_pf1 || add_s->output(0).get_target_inputs().size() != 1)
        return reject("post_ff_1 RMS");
    auto rms_post = as_rms(add_s->output(0).get_target_inputs().begin()->get_node()->output(0));
    if (!rms_post || rms_post->output(0).get_target_inputs().size() != 1)
        return reject("post_ff RMS");
    auto add2 = ov::as_type_ptr<v1::Add>(
        rms_post->output(0).get_target_inputs().begin()->get_node()->shared_from_this());
    if (!add2 || (add2->input_value(0) != h1 && add2->input_value(1) != h1))
        return reject("final residual add");

    // Optional layer_scalar multiply.
    ov::Output<ov::Node> block_out = add2->output(0);
    ov::Output<ov::Node> layer_scalar = v0::Constant::create(ov::element::f16, {1}, {1.0f})->output(0);
    if (add2->output(0).get_target_inputs().size() == 1) {
        auto mul = ov::as_type_ptr<v1::Multiply>(
            add2->output(0).get_target_inputs().begin()->get_node()->shared_from_this());
        if (mul) {
            auto c = as_constant(mul->input_value(mul->get_input_node_ptr(0) == add2.get() ? 1 : 0));
            if (c && ov::shape_size(c->get_shape()) == 1) {
                layer_scalar = mk::util::get_f16_constant(c)->output(0);
                block_out = mul->output(0);
            }
        }
    }

    // Dense MLP: down(gelu_tanh(gate(x)) * up(x)), x = RMS_pf(h1)
    auto down = rms_pf1->get_input_node_shared_ptr(0);
    auto down_w = int4_fc(down, kHidden, kInter);
    if (down_w.empty())
        return reject("down_proj");
    auto glu = ov::as_type_ptr<v1::Multiply>(down->get_input_node_shared_ptr(0));
    if (!glu)
        return reject("glu multiply");
    std::shared_ptr<ov::Node> gate, up;
    for (size_t i = 0; i < 2; ++i) {
        auto gelu = ov::as_type_ptr<ov::op::v7::Gelu>(glu->get_input_node_shared_ptr(i));
        if (gelu && gelu->get_approximation_mode() == ov::op::GeluApproximationMode::TANH) {
            gate = gelu->get_input_node_shared_ptr(0);
            up = glu->get_input_node_shared_ptr(1 - i);
        }
    }
    if (!gate)
        return reject("gelu(tanh)");
    auto gate_w = int4_fc(gate, kInter, kHidden), up_w = int4_fc(up, kInter, kHidden);
    if (gate_w.empty() || up_w.empty() || gate->input_value(0) != up->input_value(0))
        return reject("gate/up proj");
    auto rms_pf = as_rms(gate->input_value(0));
    if (!rms_pf || skip_reshape(rms_pf->input_value(0)) != h1)
        return reject("pre_ff RMS");

    ov::OutputVector in(kFfnPorts);
    in[kFfnO] = attn;
    in[kFfnOProj] = repack(oproj_w[0], oproj_w[1], oproj_w[2], kHidden, ka);
    in[kFfnH] = h;
    in[kFfnWPostAttn] = f16_gamma(rms_pa);
    in[kFfnWPreFf] = f16_gamma(rms_pf);
    in[kFfnGate] = repack(gate_w[0], gate_w[1], gate_w[2], kInter, kHidden);
    in[kFfnUp] = repack(up_w[0], up_w[1], up_w[2], kInter, kHidden);
    in[kFfnDown] = repack(down_w[0], down_w[1], down_w[2], kHidden, kInter);
    in[kFfnWPostFf1] = f16_gamma(rms_pf1);
    in[kFfnWRouter] = f16_gamma(rms_router);
    in[kFfnRouterW] = rfc->input_value(1);
    in[kFfnWPreFf2] = f16_gamma(rms_pf2);
    const auto mi = [&](int i) { return moe->input_value(3 + i); };
    in[kFfnEGate] = repack(mi(0), mi(1), mi(2), kExperts * kExpertInter, kHidden);
    in[kFfnEUp] = repack(mi(3), mi(4), mi(5), kExperts * kExpertInter, kHidden);
    in[kFfnEDown] = repack(mi(6), mi(7), mi(8), kExperts * kHidden, kExpertInter);
    in[kFfnWPostFf2] = f16_gamma(rms_pf2post);
    in[kFfnWPostFf] = f16_gamma(rms_post);
    in[kFfnLayerScalar] = layer_scalar;
    for (const auto& v : in)
        if (!v.get_node())
            return reject("gamma shape");
    if (attn.get_element_type() != ov::element::f16 || h.get_element_type() != ov::element::f16 ||
        block_out.get_element_type() != ov::element::f16 || attn.get_partial_shape().size() != 3 ||
        !attn.get_partial_shape()[2].compatible(static_cast<int64_t>(ka)) || h.get_partial_shape().size() != 3)
        return reject(("activation types attn=" + attn.get_partial_shape().to_string() +
                       attn.get_element_type().get_type_name() + " h=" + h.get_partial_shape().to_string())
                          .c_str());

    MegaKernelAttrs attrs;
    attrs.kind = kFfn;
    attrs.hidden_size = kHidden;
    attrs.intermediate_size = kInter;
    attrs.out_type = ov::element::f16;
    auto mkn = std::make_shared<MegaKernel>(in, attrs);
    mkn->set_friendly_name(block_out.get_node()->get_friendly_name() + "/MegaKernelFfn");
    ov::copy_runtime_info({oproj, moe, router, rfc, down, gate, up, add1, add2, rms_post, rms_pf1, rms_pf2post}, mkn);
    block_out.replace(mkn->output(0));
    if (dbg)
        std::cerr << "[Gemma4]   ffn " << mkn->get_friendly_name() << std::endl;
    return true;
}

int replace_ffn_blocks(const std::shared_ptr<ov::Model>& m, bool dbg) {
    const char* lim = std::getenv("OV_MEGAKERNEL_FFN_MAX");
    const int max_layers = lim ? std::atoi(lim) : std::numeric_limits<int>::max();
    int n = 0;
    for (auto& op : m->get_ordered_ops()) {
        if (n >= max_layers)
            break;
        if (auto moe = ov::as_type_ptr<MOEC>(op))
            n += replace_ffn_block(moe, dbg) ? 1 : 0;
    }
    if (dbg)
        std::cerr << "[Gemma4] FFN blocks replaced by MegaKernel: " << n << std::endl;
    return n;
}

// ---------------------------------------------------------------------------
// Pass 5: replace each layer's attention input with one MegaKernel
// ---------------------------------------------------------------------------
// MEASURED (cltrace, one decode step): input RMS -> int8 dynamic quantise ->
// oneDNN q|k|v GEMM -> f32 reorder -> q/k/v RMS -> 2x RoPE is 9 primitives and
// ~56 us of device time per sliding layer, plus a host submission gap after
// most of them. The MegaKernel runs it as two launches: a repacked int4 GEMV
// whose work-groups each recompute the input RMS (as ffn_in does), and one
// sub-group per head for the head norms and rotate-half RoPE. Its output holds
// q|k|v per token; slicing and reshaping back to heads is free.
//
// Both layer types are handled: sliding layers have one fused q|k|v matrix
// split three ways; full-attention layers have q and k projections and take v
// from k's raw output (attention_k_eq_v) through a gamma-less RMS.
struct HeadGroup {
    std::shared_ptr<RoPEOp> rope;     // q / k
    std::shared_ptr<RMSOp> rms;
    ov::Output<ov::Node> v_out;       // v: f16 tensor fed to the transpose
    int64_t heads = 0, dim = 0;
    size_t raw_off = 0;
};

ov::Output<ov::Node> skip_convert_fwd(ov::Output<ov::Node> v) {
    while (v.get_target_inputs().size() == 1) {
        auto* n = v.get_target_inputs().begin()->get_node();
        if (!ov::is_type<v0::Convert>(n))
            break;
        v = n->output(0);
    }
    return v;
}

bool is_op(const ov::Output<ov::Node>& v, const char* type) {
    return std::string(v.get_node()->get_type_name()) == type;
}

void dump_inputs(const ov::Output<ov::Node>& v, int d, int max_d);

bool replace_attn_in(const std::shared_ptr<RMSOp>& rms_in, bool dbg) {
    using ov::intel_gpu::op::MegaKernel;
    using ov::intel_gpu::op::MegaKernelAttrs;
    const bool loud = dbg && rms_in->get_friendly_name().find("input_layernorm") != std::string::npos;
    auto reject = [&](const std::string& why) {
        if (loud)
            std::cerr << "[Gemma4]   attn-skip " << rms_in->get_friendly_name() << ": " << why << std::endl;
        return false;
    };
    if (!as_rms(rms_in) || !f16_gamma(rms_in).get_node())
        return reject("input RMS");
    auto x = rms_in->input_value(0);
    if (x.get_element_type() != ov::element::f16 || x.get_partial_shape().size() != 3)
        return reject("input type");

    // Projections reading the normalised input, each split into head groups.
    std::vector<HeadGroup> roped, vs;
    std::vector<ov::OutputVector> parts;
    size_t rows = 0;
    std::vector<ov::Node*> projections;
    for (const auto& t : rms_in->output(0).get_target_inputs())
        projections.push_back(t.get_node());
    // Deterministic order: larger matrix (q or the fused q|k|v) first.
    std::sort(projections.begin(), projections.end(), [](ov::Node* a, ov::Node* b) {
        return a->get_input_partial_shape(1)[0].get_length() > b->get_input_partial_shape(1)[0].get_length();
    });
    for (ov::Node* pn : projections) {
        auto fc = pn->shared_from_this();
        if (!ov::is_type<FCC>(fc) || fc->get_input_partial_shape(1).is_dynamic() ||
            fc->get_input_shape(1).size() != 2)
            return reject("consumer is not an int4 projection");
        const size_t n = fc->get_input_shape(1)[0];
        auto w = int4_fc(fc, n, kHidden);
        if (w.empty())
            return reject("projection weights");
        parts.push_back(w);

        std::vector<std::pair<ov::Output<ov::Node>, size_t>> outs;
        auto fout = fc->output(0);
        auto split = fout.get_target_inputs().size() == 1
                         ? ov::as_type_ptr<v1::VariadicSplit>(fout.get_target_inputs().begin()->get_node()->shared_from_this())
                         : nullptr;
        if (split) {
            auto axis = as_constant(split->input_value(1));
            auto lens = as_constant(split->input_value(2));
            if (!axis || !lens || (axis->cast_vector<int64_t>()[0] != 2 && axis->cast_vector<int64_t>()[0] != -1))
                return reject("split");
            size_t off = rows;
            for (auto l : lens->cast_vector<int64_t>()) {
                outs.emplace_back(split->output(outs.size()), off);
                off += static_cast<size_t>(l);
            }
        } else {
            outs.emplace_back(fout, rows);
        }
        rows += n;

        for (auto& [out, off] : outs) {
            auto cur = skip_convert_fwd(out);
            if (cur.get_target_inputs().size() != 1)
                return reject("projection output consumers");
            auto rs = ov::as_type_ptr<v1::Reshape>(cur.get_target_inputs().begin()->get_node()->shared_from_this());
            if (!rs || rs->get_output_partial_shape(0).size() != 4 || rs->get_output_partial_shape(0)[2].is_dynamic() ||
                rs->get_output_partial_shape(0)[3].is_dynamic())
                return reject("head reshape");
            const int64_t heads = rs->get_output_partial_shape(0)[2].get_length();
            const int64_t dim = rs->get_output_partial_shape(0)[3].get_length();
            for (const auto& ti : rs->output(0).get_target_inputs()) {
                auto c = ti.get_node()->shared_from_this();
                if (ov::is_type<v0::Convert>(c)) {
                    if (c->output(0).get_target_inputs().size() != 1)
                        return reject("convert consumers");
                    c = c->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
                }
                auto r = ov::as_type_ptr<RMSOp>(c);
                if (!r || std::abs(r->get_epsilon() - 1e-6) > 1e-9 || r->output(0).get_target_inputs().size() != 1)
                    return reject("head RMS");
                HeadGroup hg;
                hg.heads = heads;
                hg.dim = dim;
                hg.raw_off = off;
                auto next = r->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
                if (auto rope = ov::as_type_ptr<RoPEOp>(next)) {
                    auto g = rms_gamma(r);
                    if (r->get_input_size() >= 2 && (!g || ov::shape_size(g->get_shape()) != static_cast<size_t>(dim)))
                        return reject("head RMS gamma");
                    hg.rope = rope;
                    hg.rms = r;
                    roped.push_back(hg);
                } else if (r->get_input_size() < 2) {
                    hg.v_out = skip_convert_fwd(r->output(0));
                    if (hg.v_out.get_element_type() != ov::element::f16)
                        return reject("v type");
                    vs.push_back(hg);
                } else {
                    return reject("RMS consumer");
                }
            }
        }
    }
    if (roped.size() != 2 || vs.size() != 1)
        return reject("head groups");
    if (roped[0].heads < roped[1].heads)
        std::swap(roped[0], roped[1]);
    const HeadGroup &q = roped[0], &k = roped[1], &v = vs[0];
    const int64_t D = q.dim;
    if (k.dim != D || v.dim != D || k.heads != v.heads || D % 16 || D > 512 ||
        q.heads + 2 * k.heads > kAttnMaxHeads || rows % 64 || rows > static_cast<size_t>(kAttnMaxRows))
        return reject("head shapes");
    const auto& qc = q.rope->get_config();
    for (const auto* hg : {&q, &k}) {
        const auto& c = hg->rope->get_config();
        if (c.is_interleaved || c.input_trans0213 || c.is_chatglm || c.support_2d_rope || c.support_3d_rope ||
            c.is_qwen || c.is_ltx_video || c.use_rope_cache || c.slice_start || c.slice_stop ||
            c.gather_position_arg_id || c.rotary_ndims != qc.rotary_ndims || c.rotary_ndims % 32 ||
            c.rotary_ndims > static_cast<size_t>(D) || hg->rope->get_input_size() != 3 ||
            hg->rope->input_value(1) != q.rope->input_value(1) || hg->rope->input_value(2) != q.rope->input_value(2))
            return reject("RoPE config");
    }
    auto cos = q.rope->input_value(1), sin = q.rope->input_value(2);
    if (std::getenv("OV_MEGAKERNEL_DUMP_ROPE"))
        dump_inputs(cos, 0, 12);
    if (cos.get_element_type() != ov::element::f16 || cos.get_partial_shape().size() != 4 ||
        cos.get_partial_shape()[3].is_dynamic() || cos.get_partial_shape()[3].get_length() < static_cast<int64_t>(qc.rotary_ndims))
        return reject("RoPE tables " + cos.get_partial_shape().to_string());

    // Tables = cos/sin(Concat(T, T)), T = Transpose(MatMul(inv_freq [1,C/2,1], pos [B,1,S])).
    // attn_post recomputes them from position_ids, so the table chain goes dead.
    auto trig = [&](ov::Output<ov::Node> v, const char* type) -> std::shared_ptr<ov::Node> {
        while (is_op(v, "Unsqueeze") || is_op(v, "Reshape") || is_op(v, "Convert"))
            v = v.get_node()->input_value(0);
        return is_op(v, type) ? v.get_node()->get_input_node_shared_ptr(0) : nullptr;
    };
    auto cat = trig(cos, "Cos");
    if (!cat || trig(sin, "Sin") != cat || !is_op(cat->output(0), "Concat") || cat->get_input_size() != 2 ||
        cat->input_value(0) != cat->input_value(1) || !is_op(cat->input_value(0), "Transpose") ||
        cat->get_output_element_type(0) != ov::element::f32)
        return reject("RoPE table concat");
    auto mm = cat->get_input_node_ptr(0)->get_input_node_shared_ptr(0);
    auto invf = mm && is_op(mm->output(0), "MatMul") ? as_constant(mm->input_value(0)) : nullptr;
    const int64_t table_c = cos.get_partial_shape()[3].get_length();
    if (!invf || invf->get_shape() != ov::Shape{1, static_cast<size_t>(table_c / 2), 1} ||
        cat->get_output_partial_shape(0)[2] != table_c)
        return reject("RoPE inv_freq");
    bool pos_f16 = false;
    ov::Output<ov::Node> pos = mm->input_value(1);
    while (pos.get_partial_shape().size() != 2 && (is_op(pos, "Convert") || is_op(pos, "Unsqueeze") || is_op(pos, "Reshape"))) {
        pos_f16 |= pos.get_element_type() == ov::element::f16;
        pos = pos.get_node()->input_value(0);
    }
    if (pos.get_partial_shape().size() != 2 ||
        (pos.get_element_type() != ov::element::i32 && pos.get_element_type() != ov::element::i64))
        return reject("RoPE positions");

    // The output is head-major [B, heads, S, D]: q and k replace RoPE outputs
    // that already were [B, heads, S, D]; v replaces its [B,S,H,D] -> [B,H,S,D]
    // transpose.
    if (!q.rope->get_config().output_trans0213 || !k.rope->get_config().output_trans0213)
        return reject("RoPE output layout");
    std::shared_ptr<ov::Node> v_transpose;
    if (v.v_out.get_target_inputs().size() == 1) {
        auto t = v.v_out.get_target_inputs().begin()->get_node()->shared_from_this();
        auto order = ov::is_type<v1::Transpose>(t) ? as_constant(t->input_value(1)) : nullptr;
        if (order && order->cast_vector<int64_t>() == std::vector<int64_t>{0, 2, 1, 3})
            v_transpose = t;
    }
    if (!v_transpose)
        return reject("v transpose");

    // Repack all projections into one row-stacked record matrix.
    auto packed = std::make_shared<v0::Constant>(ov::element::u8, ov::Shape{repacked_bytes(rows, kHidden)});
    {
        auto* dst = static_cast<uint8_t*>(const_cast<void*>(packed->get_data_ptr()));
        for (const auto& w : parts) {
            auto one = repack(w[0], w[1], w[2], w[0].get_shape()[0], kHidden);
            auto oc = ov::as_type_ptr<v0::Constant>(one.get_node_shared_ptr());
            std::memcpy(dst, oc->get_data_ptr(), oc->get_byte_size());
            dst += oc->get_byte_size();
        }
    }
    const std::vector<int32_t> meta = {static_cast<int32_t>(q.heads), static_cast<int32_t>(k.heads),
                                       static_cast<int32_t>(D), static_cast<int32_t>(qc.rotary_ndims),
                                       static_cast<int32_t>(k.raw_off), static_cast<int32_t>(v.raw_off),
                                       static_cast<int32_t>(q.raw_off),
                                       pos.get_element_type() == ov::element::i64 ? 2 : 1,
                                       pos_f16 ? 1 : 0,
                                       static_cast<int32_t>(table_c),
                                       0, 0, 0, 0, 0, 0};
    auto head_gamma = [&](const HeadGroup& hg) {
        if (auto g = rms_gamma(hg.rms))
            return mk::util::get_f16_constant(g)->output(0);
        return v0::Constant::create(ov::element::f16, {static_cast<size_t>(D)}, {1.0f})->output(0);
    };

    ov::OutputVector in(kAttnPorts);
    in[kAttnX] = x;
    in[kAttnPos] = pos;
    const auto invf_v = invf->cast_vector<float>();
    in[kAttnInvFreq] = v0::Constant::create(ov::element::f32, {invf_v.size()}, invf_v)->output(0);
    in[kAttnWIn] = f16_gamma(rms_in);
    in[kAttnQkv] = packed->output(0);
    in[kAttnWQ] = head_gamma(q);
    in[kAttnWK] = head_gamma(k);
    in[kAttnMeta] = v0::Constant::create(ov::element::i32, {meta.size()}, meta)->output(0);

    MegaKernelAttrs attrs;
    attrs.kind = kAttnIn;
    attrs.hidden_size = ov::intel_gpu::op::kMegaKernelHeadsShape;
    attrs.num_attention_heads = q.heads + 2 * k.heads;
    attrs.num_kv_heads = k.heads;
    attrs.head_dim = D;
    attrs.intermediate_size = static_cast<int64_t>(qc.rotary_ndims);
    attrs.out_type = ov::element::f16;
    auto mkn = std::make_shared<MegaKernel>(in, attrs);
    mkn->set_friendly_name(rms_in->get_friendly_name() + "/MegaKernelAttnIn");
    ov::copy_runtime_info({rms_in, q.rope, k.rope, q.rms, k.rms}, mkn);

    auto axis = v0::Constant::create(ov::element::i64, {}, {1});
    auto lens = v0::Constant::create(ov::element::i64, {3}, {q.heads, k.heads, k.heads});
    auto split = std::make_shared<v1::VariadicSplit>(mkn->output(0), axis, lens);
    // Contiguous head ranges of one buffer: the plugin turns these into in-place crops.
    q.rope->output(0).replace(split->output(0));
    k.rope->output(0).replace(split->output(1));
    v_transpose->output(0).replace(split->output(2));
    if (dbg)
        std::cerr << "[Gemma4]   attn " << mkn->get_friendly_name() << " HQ=" << q.heads << " HK=" << k.heads
                  << " D=" << D << " rn=" << qc.rotary_ndims << " rows=" << rows << " koff=" << k.raw_off
                  << " voff=" << v.raw_off << std::endl;
    return true;
}

int replace_attn_inputs(const std::shared_ptr<ov::Model>& m, bool dbg) {
    // Bisecting aid: OV_MEGAKERNEL_ATTN_ONLY=N rewrites layer N only.
    const char* only = std::getenv("OV_MEGAKERNEL_ATTN_ONLY");
    const std::string frag = only ? std::string("layers.") + only + "." : "";
    int n = 0;
    for (auto& op : m->get_ordered_ops())
        if (auto r = ov::as_type_ptr<RMSOp>(op); r && r->get_friendly_name().find("input_layernorm") != std::string::npos &&
                                                 r->get_friendly_name().find(frag) != std::string::npos)
            n += replace_attn_in(r, dbg) ? 1 : 0;
    if (dbg)
        std::cerr << "[Gemma4] attention inputs replaced by MegaKernel: " << n << std::endl;
    return n;
}

// ---------------------------------------------------------------------------
// Pass 6: build both attention masks with one MegaKernel each
// ---------------------------------------------------------------------------
// MEASURED (cltrace): the exported mask logic (vision-token groups through
// Pad/Equal/CumSum, causal triu, padding, sliding window) is ~75 tiny dynamic
// primitives, host-bound at 10-25 us each. They precede the first SDPA, so
// the GPU idles ~0.85 ms of every decode step. The rewrite matches the final
// Select/Broadcast chain, takes the constants from the graph and refuses
// anything whose structure differs from what mask_fill computes.
ov::Output<ov::Node> skip_shape_ops(ov::Output<ov::Node> v) {
    while (is_op(v, "Reshape") || is_op(v, "Unsqueeze") || is_op(v, "Squeeze") || is_op(v, "Convert"))
        v = v.get_node()->input_value(0);
    return v;
}

bool scalar_const(const ov::Output<ov::Node>& v, float& out) {
    auto c = as_constant(v);
    if (!c || ov::shape_size(c->get_shape()) != 1)
        return false;
    out = c->cast_vector<float>()[0];
    return true;
}

// Index of the input of `n` that is a scalar constant (its value in `c`), or -1.
int const_input(const ov::Node* n, float& c) {
    for (size_t i = 0; i < n->get_input_size(); ++i)
        if (scalar_const(n->input_value(i), c))
            return static_cast<int>(i);
    return -1;
}

void dump_inputs(const ov::Output<ov::Node>& v, int d, int max_d) {
    float c = 0;
    std::cerr << std::string(2 * d, ' ') << v.get_node()->get_type_name() << " " << v.get_node()->get_friendly_name()
              << " " << v.get_element_type() << v.get_partial_shape()
              << (scalar_const(v, c) ? " = " + std::to_string(c) : "") << std::endl;
    if (d < max_d && !ov::is_type<v0::Constant>(v.get_node()))
        for (const auto& in : v.get_node()->input_values())
            dump_inputs(in, d + 1, max_d);
}

int replace_attention_masks(const std::shared_ptr<ov::Model>& m, bool dbg) {
    using ov::intel_gpu::op::MegaKernel;
    using ov::intel_gpu::op::MegaKernelAttrs;
    auto reject = [&](const std::string& why) {
        if (dbg)
            std::cerr << "[Gemma4]   mask-skip: " << why << std::endl;
        return 0;
    };
    // sliding = Select(bidir, 0, Select(window, fill, full)), full = Broadcast(Add(causal, pad), shape)
    std::shared_ptr<ov::Node> sel1, sel0, full;
    for (auto& op : m->get_ordered_ops()) {
        if (!is_op(op->output(0), "Select"))
            continue;
        auto in0 = op->input_value(2);
        if (!is_op(in0, "Select") || !is_op(in0.get_node()->input_value(2), "Broadcast"))
            continue;
        sel1 = op;
        sel0 = in0.get_node_shared_ptr();
        full = sel0->get_input_node_shared_ptr(2);
        break;
    }
    if (!sel1)
        return reject("no mask chain");
    const auto out_t = sel1->get_output_element_type(0);
    if (out_t != ov::element::f16 || full->get_output_element_type(0) != out_t ||
        sel1->get_output_partial_shape(0).size() != 4)
        return reject("mask type");

    float zero1 = 1, fill = 0, fill_slide = 0, window = 0;
    if (!scalar_const(sel1->input_value(1), zero1) || zero1 != 0.0f || !scalar_const(sel0->input_value(1), fill_slide))
        return reject("select constants");
    auto win = skip_shape_ops(sel0->input_value(0)).get_node_shared_ptr();
    // q - kv; the plugin may have rewritten the Subtract as Add(q, kv * -1).
    if (!is_op(win->output(0), "GreaterEqual") || !scalar_const(win->input_value(1), window) ||
        !(is_op(win->input_value(0), "Subtract") || is_op(win->input_value(0), "Add")))
        return reject("window");

    // full = causal + pad
    auto add = full->get_input_node_shared_ptr(0);
    if (!is_op(add->output(0), "Add"))
        return reject("causal+pad add");
    std::shared_ptr<ov::Node> triu, pad;
    for (size_t i = 0; i < 2; ++i) {
        auto s = skip_shape_ops(add->input_value(i)).get_node_shared_ptr();
        if (is_op(s->output(0), "Select"))
            triu = s;
        else if (is_op(s->output(0), "Multiply") || is_op(s->output(0), "Add"))
            pad = s;
    }
    if (!triu || !pad) {
        if (dbg)
            for (size_t i = 0; i < 2; ++i)
                dump_inputs(add->input_value(i), 1, 5);
        return reject("causal/pad terms");
    }
    float zero2 = 1, causal_off = 0;
    auto bfill = triu->get_input_node_shared_ptr(1);
    if (!is_op(bfill->output(0), "Broadcast") || !scalar_const(bfill->input_value(0), fill) || fill != fill_slide ||
        !scalar_const(triu->input_value(2), zero2) || zero2 != 0.0f)
        return reject("triu select");
    auto ge = skip_shape_ops(triu->input_value(0)).get_node_shared_ptr();
    auto qrange = ge && is_op(ge->output(0), "GreaterEqual") ? skip_shape_ops(ge->input_value(1)).get_node_shared_ptr()
                                                             : nullptr;
    auto qstart = qrange && is_op(qrange->output(0), "Range") ? skip_shape_ops(qrange->input_value(0)).get_node_shared_ptr()
                                                             : nullptr;
    if (!qstart || !is_op(qstart->output(0), "Add") || const_input(qstart.get(), causal_off) < 0 ||
        !is_op(skip_shape_ops(ge->input_value(0)), "Range"))
        return reject("triu offset");

    // pad = am * p_mul + p_add, folded either way by constant folding:
    //   Add(Multiply(am, k1), k0)  or  Multiply(Subtract(c1, Multiply(am, c2)), c3)
    float p_mul = 0, p_add = 0;
    ov::Output<ov::Node> am;
    if (is_op(pad->output(0), "Add")) {
        const int ik0 = const_input(pad.get(), p_add);
        auto m1 = ik0 < 0 ? nullptr : pad->get_input_node_shared_ptr(1 - ik0);
        const int ik1 = m1 && is_op(m1->output(0), "Multiply") ? const_input(m1.get(), p_mul) : -1;
        if (ik1 < 0)
            return reject("padding term");
        am = m1->input_value(1 - ik1);
    } else {
        float c1 = 0, c2 = 0, c3 = 0;
        const int i3 = const_input(pad.get(), c3);
        auto sub = i3 < 0 ? nullptr : pad->get_input_node_shared_ptr(1 - i3);
        if (!sub || !is_op(sub->output(0), "Subtract") || !scalar_const(sub->input_value(0), c1) ||
            !is_op(sub->input_value(1), "Multiply"))
            return reject("padding term");
        auto mul = sub->get_input_node_shared_ptr(1);
        const int i2 = const_input(mul.get(), c2);
        if (i2 < 0)
            return reject("padding scale");
        am = mul->input_value(1 - i2);
        p_mul = -c2 * c3;
        p_add = c1 * c3;
    }
    // attention_mask: the first rank-2 tensor up the reshape/convert chain.
    while (am.get_partial_shape().size() != 2 &&
           (is_op(am, "Reshape") || is_op(am, "Unsqueeze") || is_op(am, "Convert")))
        am = am.get_node()->input_value(0);

    // bidir = (group_q == group_kv) & (group_kv >= 0), groups = Select(vis, cumsum(start) - 1, -1)
    auto band = skip_shape_ops(sel1->input_value(0)).get_node_shared_ptr();
    if (!is_op(band->output(0), "BitwiseAnd") && !is_op(band->output(0), "LogicalAnd"))
        return reject(std::string("vision group and: ") + band->get_type_name());
    std::shared_ptr<ov::Node> eq, nonneg;
    for (size_t i = 0; i < 2; ++i) {
        auto s = skip_shape_ops(band->input_value(i)).get_node_shared_ptr();
        if (is_op(s->output(0), "Equal"))
            eq = s;
        else if (is_op(s->output(0), "GreaterEqual"))
            nonneg = s;
    }
    float g0 = 1;
    if (!eq || !nonneg || !scalar_const(nonneg->input_value(1), g0) || g0 != 0.0f)
        return reject("vision group compare");
    auto where = skip_shape_ops(nonneg->input_value(0)).get_node_shared_ptr();
    float minus1 = 0, cum_m1 = 0;
    auto is_or = [](const ov::Output<ov::Node>& v) { return is_op(v, "BitwiseOr") || is_op(v, "LogicalOr"); };
    if (!is_op(where->output(0), "Select") || !scalar_const(where->input_value(2), minus1) || minus1 != -1.0f ||
        !is_op(where->input_value(1), "Add") || !is_op(where->get_input_node_ptr(1)->input_value(0), "CumSum") ||
        const_input(where->get_input_node_ptr(1), cum_m1) < 0 || cum_m1 != -1.0f || !is_or(where->input_value(0))) {
        if (dbg)
            dump_inputs(nonneg->input_value(0), 1, 7);
        return reject("vision groups");
    }
    auto vor = where->get_input_node_shared_ptr(0);
    float vis[2] = {0, 0};
    ov::Output<ov::Node> tti;
    for (size_t i = 0; i < 2; ++i) {
        auto e = vor->get_input_node_shared_ptr(i);
        if (!is_op(e->output(0), "Equal") || const_input(e.get(), vis[i]) != 1 || !is_op(e->input_value(0), "Pad"))
            return reject("vision ids");
        // mask_fill assumes the export's right padding of the token types.
        auto pads_begin = as_constant(e->get_input_node_ptr(0)->input_value(1));
        if (!pads_begin)
            return reject("token type padding");
        for (auto pb : pads_begin->cast_vector<int64_t>())
            if (pb != 0)
                return reject("token type padding");
        auto t = e->get_input_node_ptr(0)->input_value(0);
        if (tti.get_node() && t != tti)
            return reject("vision ids source");
        tti = t;
    }
    for (const auto& v : {am, tti})
        if (v.get_partial_shape().size() != 2 ||
            (v.get_element_type() != ov::element::i32 && v.get_element_type() != ov::element::i64))
            return reject("mask inputs " + v.get_partial_shape().to_string());

    const auto stride = [](const ov::Output<ov::Node>& v) { return v.get_element_type() == ov::element::i64 ? 2.0f : 1.0f; };
    const std::vector<float> meta = {fill, 0, p_mul, p_add, window, causal_off, vis[0], vis[1],
                                     stride(am), stride(tti), 0, 0, 0, 0, 0, 0};
    auto meta_c = v0::Constant::create(ov::element::f32, {meta.size()}, meta);
    int n = 0;
    for (const auto& [node, kind] : {std::pair<std::shared_ptr<ov::Node>, int>{full, kMaskFull}, {sel1, kMaskSliding}}) {
        MegaKernelAttrs attrs;
        attrs.kind = kind;
        attrs.hidden_size = ov::intel_gpu::op::kMegaKernelMaskShape;
        attrs.out_type = out_t;
        auto mkn = std::make_shared<MegaKernel>(ov::OutputVector{am, tti, meta_c}, attrs);
        mkn->set_friendly_name(node->get_friendly_name() + "/MegaKernelMask");
        ov::copy_runtime_info(node, mkn);
        node->output(0).replace(mkn->output(0));
        ++n;
    }
    if (dbg)
        std::cerr << "[Gemma4] attention masks replaced by MegaKernel: fill=" << fill << " pad=am*" << p_mul << "+"
                  << p_add << " window=" << window << " causal_off=" << causal_off << " vision={"
                  << vis[0] << "," << vis[1] << "} am=" << am.get_element_type() << " tti=" << tti.get_element_type()
                  << std::endl;
    return n;
}

}  // namespace

// The rewrites below are mathematically exact and would be valid anywhere, but
// iMegakernelTransformation.h requires an implementation to leave models it was
// not written for completely untouched. Gemma-4 is recognised by its 30 layers
// of paired KV state plus the post-feed-forward norm that Gemma alone has.
static bool looks_like_gemma4(const ov::NodeVector& ops) {
    if (mk::util::count_kv_read_values(ops) != 60)
        return false;
    for (const auto& op : ops) {
        if (op->get_friendly_name().find("post_feedforward_layernorm") != std::string::npos)
            return true;
    }
    return false;
}

bool InsertMegakernelTransformation(const std::shared_ptr<ov::Model>& m) {
    const bool dbg = is_dbg();
    if (!looks_like_gemma4(m->get_ordered_ops())) {
        if (dbg)
            std::cerr << "[Gemma4] not a Gemma-4 graph; pass is a no-op" << std::endl;
        return false;
    }
    // Bit mask selecting which rewrites run, for bisecting a regression:
    //   1 = gamma-less RMS fusion, 2 = sibling gamma folding,
    //   4 = post-RoPE transpose absorption, 8 = FFN-block MegaKernel,
    //   16 = attention-input MegaKernel, 32 = attention-mask MegaKernel.
    const char* mask_env = std::getenv("OV_MEGAKERNEL_PASSES");
    const int mask = mask_env ? std::atoi(mask_env) : 63;

    bool changed = false;
    if (mask & 1) {
        int gammaless = 0;
        ov::pass::Manager mgr("Gemma4MegaKernel");
        mgr.register_pass<FuseGammalessRMS>(gammaless);
        changed |= mgr.run_passes(m);
        if (dbg)
            std::cerr << "[Gemma4] gamma-less RMS chains fused: " << gammaless << std::endl;
    }
    if (mask & 2)
        changed |= fold_sibling_rms_gamma(m, dbg) > 0;
    if (mask & 4)
        changed |= fold_transpose_into_rope(m, dbg) > 0;
    if (mask & 8)
        changed |= replace_ffn_blocks(m, dbg) > 0;
    if (mask & 16)
        changed |= replace_attn_inputs(m, dbg) > 0;
    if (mask & 32)
        changed |= replace_attention_masks(m, dbg) > 0;

    if (const char* opt_in = std::getenv("OV_MEGAKERNEL_LMHEAD"); opt_in && opt_in[0] == '1') {
        if (replace_lm_head(m, dbg)) {
            changed = true;
            m->validate_nodes_and_infer_types();
            mk::util::verify_and_dump(m, 60, 1, "MegaKernelLMHead");
        }
    }

    if (changed)
        m->validate_nodes_and_infer_types();

    if (const char* d = std::getenv("OV_MEGAKERNEL_DUMP_LAYER")) {
        const std::string frag = std::string("layers.") + d + ".";
        const std::string frag2 = std::string("layers.") + d + "/";
        auto in_layer = [&](const std::string& n) {
            return n.find(frag) != std::string::npos || n.find(frag2) != std::string::npos;
        };
        for (auto& op : m->get_ordered_ops()) {
            const auto& n = op->get_friendly_name();
            bool feeds = false;
            for (auto& t : op->output(0).get_target_inputs())
                feeds |= in_layer(t.get_node()->get_friendly_name());
            if (!in_layer(n) && !feeds)
                continue;
            std::cerr << "[Gemma4] " << op->get_type_name() << " " << n << " -> " << op->get_output_element_type(0)
                      << op->get_output_partial_shape(0);
            if (auto c = ov::as_type_ptr<v0::Constant>(op); c && ov::shape_size(c->get_shape()) <= 4 &&
                                                          c->get_element_type().is_real()) {
                std::cerr << " = {";
                for (auto v : c->cast_vector<float>())
                    std::cerr << v << " ";
                std::cerr << "}";
            }
            if (auto r = ov::as_type_ptr<RMSOp>(op))
                std::cerr << " eps=" << r->get_epsilon();
            if (auto mo = ov::as_type_ptr<ov::op::internal::MOECompressed>(op)) {
                const auto& c = mo->get_config();
                std::cerr << " H=" << c.hidden_size << " I=" << c.inter_size << " E=" << c.num_expert
                          << " topk=" << c.top_k << " gs=" << c.group_size << " zp=" << c.has_zp
                          << " act=" << static_cast<int>(c.activation_type)
                          << " sf=" << c.scale_factor.value_or(-1.0f) << " batchdim=" << c.has_batch_dim;
            }
            std::cerr << "\n";
            for (size_t i = 0; i < op->get_input_size(); ++i) {
                auto src = op->input_value(i);
                std::cerr << "      in" << i << ": " << src.get_node()->get_type_name() << " "
                          << src.get_node()->get_friendly_name() << ":" << src.get_index() << " "
                          << src.get_element_type() << src.get_partial_shape() << "\n";
            }
        }
    }
    return changed;
}

}  // namespace mk

