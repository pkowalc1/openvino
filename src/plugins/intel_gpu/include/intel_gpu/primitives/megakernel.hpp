// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

struct megakernel : public primitive_base<megakernel> {
    CLDNN_DECLARE_PRIMITIVE(megakernel)

    megakernel() : primitive_base("", {}) {}

    megakernel(const primitive_id& id,
                      const std::vector<input_info>& inputs,
                      int64_t num_layers,
                      int64_t hidden_size,
                      int64_t num_kv_heads,
                      int64_t head_dim,
                      int64_t num_heads,
                      int64_t intermediate_size,
                      float   rms_norm_eps)
        : primitive_base(id, inputs, 1 /*num_outputs: hidden_states only; KV cache is internal*/),
          num_layers(num_layers),
          hidden_size(hidden_size),
          num_kv_heads(num_kv_heads),
          head_dim(head_dim),
          num_heads(num_heads),
          intermediate_size(intermediate_size),
          rms_norm_eps(rms_norm_eps) {}

    int64_t num_layers  = 28;
    int64_t hidden_size = 1024;
    int64_t num_kv_heads = 8;
    int64_t head_dim    = 128;
    int64_t num_heads   = 16;
    int64_t intermediate_size = 3072;
    float   rms_norm_eps = 1e-6f;
    int64_t kind = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, kind);
        seed = hash_combine(seed, num_layers);
        seed = hash_combine(seed, hidden_size);
        seed = hash_combine(seed, num_kv_heads);
        seed = hash_combine(seed, head_dim);
        seed = hash_combine(seed, num_heads);
        seed = hash_combine(seed, intermediate_size);
        seed = hash_combine(seed, rms_norm_eps);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_c = downcast<const megakernel>(rhs);
        return kind == rhs_c.kind &&
               num_layers   == rhs_c.num_layers &&
               hidden_size  == rhs_c.hidden_size &&
               num_kv_heads == rhs_c.num_kv_heads &&
               head_dim     == rhs_c.head_dim &&
               num_heads    == rhs_c.num_heads &&
               intermediate_size == rhs_c.intermediate_size &&
               rms_norm_eps == rhs_c.rms_norm_eps;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<megakernel>::save(ob);
        ob << num_layers << hidden_size << num_kv_heads << head_dim
           << num_heads << intermediate_size << rms_norm_eps << kind;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<megakernel>::load(ib);
        ib >> num_layers >> hidden_size >> num_kv_heads >> head_dim
           >> num_heads >> intermediate_size >> rms_norm_eps >> kind;
    }
};

}  // namespace cldnn
