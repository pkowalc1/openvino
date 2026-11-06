// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/megakernel.hpp>

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/op/megakernel.hpp"
#include "megakernel_inst.h"
#include "openvino/op/parameter.hpp"
#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;
using ov::intel_gpu::op::MegaKernel;
using ov::intel_gpu::op::MegaKernelAttrs;
using ov::intel_gpu::op::MegaKernelDim;

namespace shape_infer_tests {
namespace {

// Two outputs: [in0.0, in0.1, 64] f16 and a 5D [in1.0, 2, in0.1, in1.1, 8] i32, plus some params.
MegaKernelAttrs make_attrs() {
    MegaKernelAttrs attrs;
    attrs.outputs = {
        {ov::element::f16, {MegaKernelDim::of_input(0, 0), MegaKernelDim::of_input(0, 1), MegaKernelDim::constant(64)}},
        {ov::element::i32,
         {MegaKernelDim::of_input(1, 0), MegaKernelDim::constant(2), MegaKernelDim::of_input(0, 1), MegaKernelDim::of_input(1, 1),
          MegaKernelDim::constant(8)}},
    };
    attrs.params = {1, 2, 3, 0, 255};
    return attrs;
}

// Reads the attributes of an op, or writes them into it when `write` is set.
struct AttrsVisitor : public ov::AttributeVisitor {
    bool write = false;
    ov::element::TypeVector output_types;
    std::vector<int64_t> output_shapes;
    std::vector<uint8_t> params;

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        auto* a = ov::as_type<ov::AttributeAdapter<ov::element::TypeVector>>(&adapter);
        ASSERT_NE(a, nullptr) << name;
        write ? a->set(output_types) : void(output_types = a->get());
    }
    void on_adapter(const std::string&, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        write ? adapter.set(output_shapes) : void(output_shapes = adapter.get());
    }
    void on_adapter(const std::string&, ov::ValueAccessor<std::vector<uint8_t>>& adapter) override {
        write ? adapter.set(params) : void(params = adapter.get());
    }
};

}  // namespace

TEST(megakernel_si_test, calc_output_layouts) {
    auto& engine = get_test_engine();
    const layout in0{ov::PartialShape{1, ov::Dimension::dynamic(), 16}, data_types::f16, format::bfyx};
    const layout in1{ov::PartialShape{3, 7}, data_types::i64, format::bfyx};

    auto mk = std::make_shared<megakernel>("mk", std::vector<input_info>{input_info("in0"), input_info("in1")}, make_attrs());

    cldnn::program prog(engine);
    auto& in0_node = prog.get_or_create(std::make_shared<input_layout>("in0", in0));
    auto& in1_node = prog.get_or_create(std::make_shared<input_layout>("in1", in1));
    auto& mk_node = prog.get_or_create(mk);
    program_wrapper::add_connection(prog, in0_node, mk_node);
    program_wrapper::add_connection(prog, in1_node, mk_node);

    auto res = megakernel_inst::calc_output_layouts<ov::PartialShape>(mk_node, *mk_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 2);
    ASSERT_EQ(res[0], (layout{ov::PartialShape{1, ov::Dimension::dynamic(), 64}, data_types::f16, format::bfyx}));
    ASSERT_EQ(res[1], (layout{ov::PartialShape{3, 2, ov::Dimension::dynamic(), 7, 8}, data_types::i32, format::bfzyx}));
}

TEST(megakernel_si_test, op_infers_clones_and_visits_attributes) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, ov::Dimension::dynamic(), 16});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    auto op = std::make_shared<MegaKernel>(ov::OutputVector{in0, in1}, make_attrs());

    ASSERT_EQ(op->get_output_size(), 2);
    ASSERT_EQ(op->get_output_element_type(0), ov::element::f16);
    ASSERT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{1, ov::Dimension::dynamic(), 64}));
    ASSERT_EQ(op->get_output_element_type(1), ov::element::i32);
    ASSERT_EQ(op->get_output_partial_shape(1), (ov::PartialShape{ov::Dimension::dynamic(), 2, ov::Dimension::dynamic(), ov::Dimension::dynamic(), 8}));

    auto clone = std::dynamic_pointer_cast<MegaKernel>(op->clone_with_new_inputs({in0, in1}));
    ASSERT_TRUE(clone->get_attrs() == make_attrs());

    AttrsVisitor visitor;
    op->visit_attributes(visitor);
    auto restored = std::make_shared<MegaKernel>();
    visitor.write = true;
    restored->visit_attributes(visitor);
    ASSERT_TRUE(restored->get_attrs() == make_attrs());
}

TEST(megakernel_si_test, primitive_compares_and_serializes_attributes) {
    const megakernel prim("mk", {input_info("in0"), input_info("in1")}, make_attrs());

    auto other_attrs = make_attrs();
    other_attrs.params.back() = 7;
    const megakernel other("mk", {input_info("in0"), input_info("in1")}, other_attrs);
    ASSERT_FALSE(prim == other);
    ASSERT_NE(prim.hash(), other.hash());

    std::stringstream ss;
    BinaryOutputBuffer ob(ss);
    prim.save(ob);
    megakernel loaded;
    BinaryInputBuffer ib(ss, get_test_engine());
    loaded.load(ib);
    ASSERT_TRUE(loaded.attrs == make_attrs());
    ASSERT_TRUE(loaded == prim);
}

}  // namespace shape_infer_tests
