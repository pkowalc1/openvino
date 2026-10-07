#pragma once

#include <memory>

namespace ov {
class Model;
}

namespace mk {

// Graph transformation interface for MEGAKERNEL POC.
//
// Every megakernel implementation directory must provide a `transformation/`
// sub-directory with exactly one definition of this function. Unlike the
// runtime (which is a stand-alone shared library, see iMegakernelRuntime.h),
// the transformation is compiled directly into the GPU plugin: it needs the
// OpenVINO opset and the plugin-internal `ov::intel_gpu::op::MegaKernel` op,
// so shipping it as a separate library would create a circular dependency.
//
// The sources of the implementation selected by the MEGAKERNEL_IMPLEMENTATION
// CMake cache variable are appended to the plugin's source list, so exactly one
// definition ends up in the final binary.
//
// Returns true if the model was modified (i.e. the megakernel was inserted).
// Implementations must return false - without touching the model - whenever the
// graph does not match the architecture they were written for, so that
// unrelated models keep compiling normally.
bool InsertMegakernelTransformation(const std::shared_ptr<ov::Model>& model);

}  // namespace mk
