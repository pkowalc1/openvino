#pragma once
#include "../iMegakernelBinding.h"
#include "../iMegakernelRuntime.h"
#include "exportApi.h"
#include "gemma4POCParams.h"

extern "C" EXPORT_API mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime();

extern "C" EXPORT_API void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime);

// Port order is defined by transformation/gemma4POCTransformation.cpp.
extern "C" EXPORT_API void FillMegaKernelConstantParams(mk::IConstantParams* params,
                                                        const mk::MegakernelIo* io);

extern "C" EXPORT_API void FillMegaKernelRuntimeParams(mk::IRuntimeParams* params,
                                                       const mk::MegakernelIo* io);
