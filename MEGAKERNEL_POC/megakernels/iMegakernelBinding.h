#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "iMegakernelRuntime.h"

namespace mk {

// Lazy accessors to the MegaKernel op's graph ports
struct MegakernelIo {
  void* ctx;
  void* (*input_ptr)(void* ctx, int port);
  void* (*output_ptr)(void* ctx, int port);
  int64_t (*input_dim)(void* ctx, int port, int axis);
  int64_t (*output_dim)(void* ctx, int port, int axis);
  const void* params;
  size_t params_size;
};

template <typename T>
bool ReadParams(const MegakernelIo& io, T& out) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (io.params_size != sizeof(T)) return false;
  std::memcpy(&out, io.params, sizeof(T));
  return true;
}

}  // namespace mk

// The port order and params are defined by the megakernel's own transformation,
// so mapping them onto its param structs belongs to the megakernel too. Every
// megakernel exports these from megakernelImpl.h; non-zero return is an error:
//
//   extern "C" EXPORT_API mk::TErrorcode FillMegaKernelConstantParams(
//       mk::IConstantParams* params, const mk::MegakernelIo* io);
//   extern "C" EXPORT_API mk::TErrorcode FillMegaKernelRuntimeParams(
//       mk::IRuntimeParams* params, const mk::MegakernelIo* io);
