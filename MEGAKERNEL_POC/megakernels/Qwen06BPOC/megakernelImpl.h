#pragma once
#include "../iMegakernelRuntime.h"
#include "exportApi.h"
#include "qwen06BPOCParams.h"

// Create a MegaKernelPOCRuntime instance.
extern "C" EXPORT_API mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime();

// Destroy a MegaKernelPOCRuntime instance.
extern "C" EXPORT_API void DestroyMegaKernelPOCRuntime(
    mk::IMegakernelRuntime* runtime);
