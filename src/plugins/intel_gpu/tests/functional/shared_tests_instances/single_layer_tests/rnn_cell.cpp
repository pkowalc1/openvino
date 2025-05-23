// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/rnn_cell.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::RNNCellTest;

namespace {
    std::vector<bool> should_decompose{false, true};
    std::vector<size_t> batch{1, 5};
    std::vector<size_t> hidden_size{1, 10};
    std::vector<size_t> input_size{1, 30};
    std::vector<std::vector<std::string>> activations = {{"relu"}, {"sigmoid"}, {"tanh"}};
    std::vector<float> clip = {0.f, 0.7f};
    std::vector<ov::test::utils::InputLayerType> layer_types = {
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::PARAMETER
    };
    std::vector<ov::element::Type> model_types = {ov::element::f32,
                                                  ov::element::f16};

    INSTANTIATE_TEST_SUITE_P(smoke_RNNCellCommon1, RNNCellTest,
            ::testing::Combine(
            ::testing::ValuesIn(should_decompose),
            ::testing::ValuesIn(batch),
            ::testing::ValuesIn(hidden_size),
            ::testing::ValuesIn(input_size),
            ::testing::ValuesIn(activations),
            ::testing::Values(clip[0]),
            ::testing::Values(layer_types[0]),
            ::testing::Values(layer_types[0]),
            ::testing::Values(layer_types[0]),
            ::testing::Values(model_types[0]),
            ::testing::Values(ov::test::utils::DEVICE_GPU)),
            RNNCellTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_RNNCellCommon2, RNNCellTest,
            ::testing::Combine(
            ::testing::Values(should_decompose[0]),
            ::testing::Values(batch[0]),
            ::testing::Values(hidden_size[0]),
            ::testing::Values(input_size[0]),
            ::testing::Values(activations[0]),
            ::testing::ValuesIn(clip),
            ::testing::ValuesIn(layer_types),
            ::testing::ValuesIn(layer_types),
            ::testing::ValuesIn(layer_types),
            ::testing::ValuesIn(model_types),
            ::testing::Values(ov::test::utils::DEVICE_GPU)),
            RNNCellTest::getTestCaseName);

}  // namespace
