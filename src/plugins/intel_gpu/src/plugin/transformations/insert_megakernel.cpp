// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Thin dispatcher for the MegaKernel graph transformation.
//
// The transformation itself is model-specific and lives next to the megakernel
// it belongs to, in MEGAKERNEL_POC/megakernels/<Name>/transformation/. The
// implementation selected by the MEGAKERNEL_IMPLEMENTATION CMake cache variable
// is compiled into this plugin and provides the single definition of
// mk::InsertMegakernelTransformation (see megakernels/iMegakernelTransformation.h).

#include "insert_megakernel.hpp"

#include "iMegakernelTransformation.h"

#include <cstdlib>

namespace ov::intel_gpu {

bool InsertMegaKernel::run_on_model(const std::shared_ptr<ov::Model>& m) {
    if (const char* off = std::getenv("OV_MEGAKERNEL_DISABLE"); off && off[0] == '1')
        return false;

    return mk::InsertMegakernelTransformation(m);
}

}  // namespace ov::intel_gpu
