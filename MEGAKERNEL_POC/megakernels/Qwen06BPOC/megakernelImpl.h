#pragma once
#include "../iMegakernelBinding.h"
#include "../iMegakernelRuntime.h"
#include "exportApi.h"
#include "qwen06BPOCParams.h"

// Create a MegaKernelPOCRuntime instance.
extern "C" EXPORT_API mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime();

// Destroy a MegaKernelPOCRuntime instance.
extern "C" EXPORT_API void DestroyMegaKernelPOCRuntime(
    mk::IMegakernelRuntime* runtime);

// Map graph ports onto this megakernel's parameter structs. The port order is
// defined by this megakernel's own transformation (transformation/*.cpp).
extern "C" EXPORT_API void FillMegaKernelConstantParams(
    mk::IConstantParams* params, const mk::MegakernelIo* io);

extern "C" EXPORT_API void FillMegaKernelRuntimeParams(
    mk::IRuntimeParams* params, const mk::MegakernelIo* io);
