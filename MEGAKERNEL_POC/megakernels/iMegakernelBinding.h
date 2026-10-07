#pragma once

#include <cstdint>

#include "iMegakernelRuntime.h"

namespace mk {

// Abstracts the plugin's cldnn::primitive_inst away from megakernel-specific
// binding code: a megakernel only needs raw USM pointers for its graph ports
// plus a few shape values. The plugin fills this in and hands it over.
struct MegakernelIo {
  void* ctx;
  void* (*input_ptr)(void* ctx, int port);
  void* (*output_ptr)(void* ctx, int port);
  int64_t (*input_dim)(void* ctx, int port, int axis);
  // MegaKernelAttrs::kind of the op being bound.
  int64_t kind;
};

}  // namespace mk

// Which graph port carries which weight is decided by the megakernel's own
// transformation, so the mapping back onto ConstantParams/RuntimeParams belongs
// to the megakernel too - otherwise every new model would need an edit inside
// the plugin. Both functions are exported by the megakernel shared library and
// declared alongside Create/DestroyMegaKernelPOCRuntime in megakernelImpl.h.
//
//   extern "C" EXPORT_API void FillMegaKernelConstantParams(
//       mk::IConstantParams* params, const mk::MegakernelIo* io);
//   extern "C" EXPORT_API void FillMegaKernelRuntimeParams(
//       mk::IRuntimeParams* params, const mk::MegakernelIo* io);
