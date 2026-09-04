// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MegaKernel plugin implementation — task-system-scheduled decoder.
// The entire Qwen3 decoder (all layers) runs in ONE kernel launch per token, but
// instead of a persistent grid + grid-wide software barrier, the work is driven
// by the fine-tuned GPU task system (MEGAKERNEL_POC/research/preloading_gemv):
// a pool of persistent worker work-groups pulls topologically-sorted tasks FIFO
// from a shared work queue; each layer stage is a set of per-workgroup tiles, and
// inter-stage ordering (the old grid barrier) is expressed as global atomic
// sync-flag dependencies resolved by the tasks themselves. Key techniques:
// intel_sub_group_block_read, fused RMSNorm, fused RoPE, workgroup-cooperative
// flash-decoding attention, split-K GEMV.
//
// This impl only ever sees DECODE steps: InsertMegaKernel routes prompts to the
// original OpenVINO sub-graph and feeds this op a zero-length tensor during
// prefill, which makes cldnn skip it entirely. The KV cache produced by that
// prefill is handed over here — it is imported once into the runtime's internal
// cache, which then owns the sequence for the rest of the generation.

#include "megakernel.hpp"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "../primitive_ocl_base.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/primitives/megakernel.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "megakernelImpl.h"
#include "megakernel_inst.h"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_event.hpp"
#include "ocl/ocl_stream.hpp"

namespace ov::intel_gpu::ocl {

using cldnn::ocl::ocl_engine;
using cldnn::ocl::ocl_event;
using cldnn::ocl::ocl_stream;

namespace {
// ---------------------------------------------------------------------------
// MegaKernelFastImpl
// ---------------------------------------------------------------------------

class MegaKernelFastImpl : public cldnn::primitive_impl {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MegaKernelFastImpl)

    MegaKernelFastImpl() = default;
    explicit MegaKernelFastImpl(const cldnn::program_node&, const RuntimeParams&) {}
    // Copy constructor: copy the primitive_impl base subobject so metadata such
    // as m_manager and the dynamic flag are preserved (required by the impl
    // caches in ImplementationsFactory). The OpenCL/runtime members below keep
    // their default null/zero initializers so device state is re-created lazily.
    MegaKernelFastImpl(const MegaKernelFastImpl& other) : cldnn::primitive_impl(other) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> clone() const override {
        OPENVINO_ASSERT(megakernelRuntime_ == nullptr,
                        "[GPU] MegaKernelFastImpl::clone() should not be called if megakernel runtime is initialized; use create_impl() instead.");
        return std::make_unique<MegaKernelFastImpl>(*this);
    }
    bool is_cpu() const override {
        return false;
    }
    void save(BinaryOutputBuffer&) const override {}
    void load(BinaryInputBuffer&) override {}
    void init_kernels(const cldnn::kernels_cache&, const cldnn::kernel_impl_params&) override {}
    void set_arguments(cldnn::primitive_inst&) override {}
    void set_arguments(cldnn::primitive_inst&, cldnn::kernel_arguments_data&) override {}
    std::vector<cldnn::BufferDescriptor> get_internal_buffer_descs(const cldnn::kernel_impl_params&) const override {
        return {};
    }

    void ensure_ready(cldnn::primitive_inst& instance) {
        std::lock_guard<std::mutex> g(mu_);
        if (megakernelRuntime_ != nullptr) {
            return;
        }

        megakernelRuntime_ = CreateMegaKernelPOCRuntime();

        auto& eng = cldnn::downcast<cldnn::ocl::ocl_engine>(instance.get_network().get_engine());
        cl_context ctx = eng.get_cl_context().get();
        cl_device_id dl_device = eng.get_cl_device().get();
        cl_command_queue queue = cldnn::downcast<cldnn::ocl::ocl_stream>(instance.get_network().get_stream()).get_cl_queue().get();

        // Resolve the raw USM device pointers for every model input/output. The
        // task-system tasks dereference these directly out of the context struct,
        // which requires genuine USM device allocations (asserted below).
        auto usm_raw = [](cldnn::memory& m, const char* name) -> void* {
            auto at = m.get_allocation_type();
            bool usm = at == cldnn::allocation_type::usm_device || at == cldnn::allocation_type::usm_host || at == cldnn::allocation_type::usm_shared;
            OPENVINO_ASSERT(usm, "[MegaKernel] input/output '", name, "' must be a USM allocation for the task-system path");
            return m.buffer_ptr();
        };

        // TODO: Specific to Qwen06BPOC, but will have to be generalized to any megakernel runtime with a known constant parameter type.
        mk::ConstantParamsImpl weights{};
        weights.q_proj_w = usm_raw(instance.input_memory(3), "q_proj_w");
        weights.k_proj_w = usm_raw(instance.input_memory(4), "k_proj_w");
        weights.v_proj_w = usm_raw(instance.input_memory(5), "v_proj_w");
        weights.o_proj_w = usm_raw(instance.input_memory(6), "o_proj_w");
        weights.gate_proj_w = usm_raw(instance.input_memory(7), "gate_proj_w");
        weights.up_proj_w = usm_raw(instance.input_memory(8), "up_proj_w");
        weights.down_proj_w = usm_raw(instance.input_memory(9), "down_proj_w");
        weights.input_ln_w = usm_raw(instance.input_memory(10), "input_ln_w");
        weights.post_attn_ln_w = usm_raw(instance.input_memory(11), "post_attn_ln_w");
        weights.q_norm_w = usm_raw(instance.input_memory(12), "q_norm_w");
        weights.k_norm_w = usm_raw(instance.input_memory(13), "k_norm_w");
        weights.rope_inv_freq = usm_raw(instance.input_memory(14), "rope_inv_freq");
        // ---

        // Specific to OpenCL platform, but will have to be generalized to any plaform supported by megakernel runtime.
        mk::PlatformParamsImpl platformParams{};
        platformParams.context = ctx;
        platformParams.deviceId = dl_device;
        platformParams.stream = queue;
        megakernelRuntime_->Init(&weights, &platformParams);
    }

    // Resolve the OpenVINO KV-cache variables (one Key/Value pair per layer) that the
    // prefill path fills in. Only the variable *ids* are stable: every infer request
    // owns its own VariableState objects and rebinds them on the network before each
    // inference, so the handles have to be fetched again on every call.
    void resolve_kv_variables(cldnn::primitive_inst& instance) {
        auto& net = instance.get_network();
        const auto num_layers = static_cast<size_t>(instance.get_impl_params()->typed_desc<cldnn::megakernel>()->num_layers);

        if (kv_key_ids_.empty()) {
            std::map<int, std::string> keys, values;
            for (const auto& [id, var] : net.get_variables()) {
                const auto pos = id.find("past_key_values.");
                if (pos == std::string::npos)
                    continue;
                const auto num_start = pos + std::strlen("past_key_values.");
                const auto num_end = id.find('.', num_start);
                if (num_end == std::string::npos)
                    continue;
                const int layer = std::stoi(id.substr(num_start, num_end - num_start));
                (id.compare(num_end + 1, 3, "key") == 0 ? keys : values)[layer] = id;
            }
            OPENVINO_ASSERT(keys.size() == num_layers && values.size() == num_layers,
                            "[MegaKernel] expected ", num_layers, " key/value cache variables, found ",
                            keys.size(), "/", values.size());
            for (size_t l = 0; l < num_layers; ++l) {
                kv_key_ids_.push_back(keys.at(static_cast<int>(l)));
                kv_value_ids_.push_back(values.at(static_cast<int>(l)));
            }
        }

        kv_key_vars_.clear();
        kv_value_vars_.clear();
        for (size_t l = 0; l < num_layers; ++l) {
            kv_key_vars_.push_back(&net.get_variable(kv_key_ids_[l]));
            kv_value_vars_.push_back(&net.get_variable(kv_value_ids_[l]));
        }

        // A single impl -- and therefore a single megakernel runtime with a single
        // internal KV cache -- is shared by every infer request of the compiled
        // model. Switching request invalidates the internal cache, so force a fresh
        // import. Requests must be used sequentially: interleaving them re-imports on
        // every step, which stays correct but is slow. (PoC limitation.)
        if (kv_key_vars_[0] != owner_) {
            owner_ = kv_key_vars_[0];
            cache_len_ = -1;
            last_kv_len_ = -1;
        }
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        ensure_ready(instance);
        resolve_kv_variables(instance);

        auto& strm = instance.get_network().get_stream();
        auto& ocls = downcast<ocl_stream>(strm);
        cl_command_queue q = ocls.get_cl_queue().get();

        for (auto& e : events)
            strm.wait_for_events({e});  // inputs ready before we read them

        OPENVINO_ASSERT(instance.input_memory(1).get_layout().data_type == cldnn::data_types::i64,
                        "[MegaKernel] supports only i64 position_ids (input 1) for the task-system path");

        // The position of the token we are about to generate, i.e. how many tokens
        // must already be present in the KV cache.
        int64_t pos = 0;
        {
            cldnn::mem_lock<int64_t, cldnn::mem_lock_type::read> lock(instance.input_memory_ptr(1), strm);
            pos = lock[0];
        }

        // The OpenVINO KV cache only grows during prefill (the original path sees
        // zero tokens while decoding), so a change of its length marks a new prompt.
        // pos == cache_len_ is the normal next step and pos == cache_len_ - 1 is a
        // replay of the previous step (used by the decode microbenchmark); anything
        // else means our internal cache does not cover [0, pos) and must be reloaded.
        const auto& key_layout = kv_key_vars_[0]->get_layout();
        const bool has_prefill_kv = key_layout.get_partial_shape().is_static();
        const int64_t kv_len = has_prefill_kv ? static_cast<int64_t>(key_layout.get_shape()[2]) : 0;
        const bool import_past =
            has_prefill_kv && ((kv_len != last_kv_len_) || (pos != cache_len_ && pos + 1 != cache_len_));

        std::vector<const void*> past_key, past_value;
        std::vector<int> past_key_stride, past_value_stride;
        if (import_past) {
            const auto num_layers = kv_key_vars_.size();
            past_key.reserve(num_layers);
            past_value.reserve(num_layers);
            past_key_stride.reserve(num_layers);
            past_value_stride.reserve(num_layers);
            // Every cache tensor is preallocated independently, so each one has its
            // own sequence stride; reading them all with a single stride corrupts
            // everything but layer 0's keys.
            auto collect = [&](ov::intel_gpu::VariableStateBase* var, std::vector<const void*>& ptrs, std::vector<int>& strides) {
                const auto& lay = var->get_layout();
                OPENVINO_ASSERT(lay.data_type == cldnn::data_types::f16,
                                "[MegaKernel] only an f16 KV cache can be imported from the prefill path");
                OPENVINO_ASSERT(static_cast<int64_t>(lay.get_shape()[2]) == kv_len,
                                "[MegaKernel] inconsistent KV cache lengths across layers");
                ptrs.push_back(var->get_memory()->buffer_ptr());
                strides.push_back(lay.get_padded_dims()[2]);
            };
            for (size_t l = 0; l < num_layers; ++l) {
                collect(kv_key_vars_[l], past_key, past_key_stride);
                collect(kv_value_vars_[l], past_value, past_value_stride);
            }
        }

        if (const char* dbg = std::getenv("OV_MEGAKERNEL_DEBUG"); dbg && dbg[0] == '1') {
            std::cout << "[MegaKernel] pos=" << pos << " kv_len=" << kv_len << " cache_len=" << cache_len_
                      << " import=" << import_past << " key_layout=" << key_layout.to_short_string() << std::endl;
        }

        // TODO: Specific to Qwen06BPOC, but will have to be generalized to any megakernel runtime with a known runtime parameter type.
        mk::RuntimeParamsImpl io{};
        io.hidden_states = instance.input_memory(0).buffer_ptr();
        io.position_ids = instance.input_memory(1).buffer_ptr();
        io.hidden_states_out = instance.output_memory(0).buffer_ptr();
        io.newTokens = (int)instance.input_memory(0).get_layout().get<ov::PartialShape>()[1].get_length();
        if (import_past) {
            io.past_key = past_key.data();
            io.past_value = past_value.data();
            io.past_key_stride = past_key_stride.data();
            io.past_value_stride = past_value_stride.data();
            io.past_len = static_cast<int>(kv_len);
            io.import_past = true;
        }
        // ---

        megakernelRuntime_->Execute(&io);

        last_kv_len_ = kv_len;
        cache_len_ = std::max(import_past ? kv_len : cache_len_, pos + 1);

        cl_event marker;
        clEnqueueMarkerWithWaitList(q, 0, nullptr, &marker);
        return std::make_shared<ocl_event>(cl::Event(marker, false), 0ULL);
    }

    ~MegaKernelFastImpl() override {
        if (megakernelRuntime_)
            DestroyMegaKernelPOCRuntime(megakernelRuntime_);
        megakernelRuntime_ = nullptr;
    }

private:
    std::mutex mu_;
    mk::IMegakernelRuntime* megakernelRuntime_ = nullptr;
    std::vector<std::string> kv_key_ids_, kv_value_ids_;
    std::vector<ov::intel_gpu::VariableStateBase*> kv_key_vars_;
    std::vector<ov::intel_gpu::VariableStateBase*> kv_value_vars_;
    // The infer request (identified by its layer-0 key state) the counters below
    // belong to.
    const ov::intel_gpu::VariableStateBase* owner_ = nullptr;
    // Number of leading positions our internal KV cache holds, and the length of the
    // OpenVINO cache the last import was taken from.
    int64_t cache_len_ = -1;
    int64_t last_kv_len_ = -1;
};

}  // namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
std::unique_ptr<cldnn::primitive_impl> MegaKernelImpl::create_impl(const cldnn::program_node& node, const RuntimeParams& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::megakernel>());
    return std::make_unique<MegaKernelFastImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::megakernel)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MegaKernelFastImpl)