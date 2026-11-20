#include "megakernelImpl.h"

#include "impl/qwen06BPOCRuntime.h"

/////////////////////////////////////////////////////////////////////////
extern "C" mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime() {
  return new mk::Qwen06BPOCRuntime();
}

/////////////////////////////////////////////////////////////////////////
extern "C" void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime) {
  if (runtime) delete runtime;
}
