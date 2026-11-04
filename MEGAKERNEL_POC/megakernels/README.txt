Folder contains POC impl of megakernel implemenations for openvino.

Currrently the idea is that each megakernel will be delivered as a stand alone library,
containing kernel plus runtime implementation.

The openvino engine has to link against that library.

Each implementation is backed by header-only, interface lib megakernel runtime, which defines 
interface that openvino engien will link against. 

All megakernels implementations has to export functions:
-> extern "C" mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime();
-> extern "C" void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime);
-> extern "C" mk::TErrorcode FillMegaKernelConstantParams(mk::IConstantParams*, const mk::MegakernelIo*);
-> extern "C" mk::TErrorcode FillMegaKernelRuntimeParams(mk::IRuntimeParams*, const mk::MegakernelIo*);
which has to be defined in file megakernelImpl.h (see iMegakernelBinding.h)

Create/Destroy create the runtime for given megakernel; Fill* map the MegaKernel
op's graph ports and params onto its param structs (non-zero return = error).

Each implementation also has to provide transformation/*.cpp defining
mk::InsertMegakernelTransformation (see iMegakernelTransformation.h), which inserts
the MegaKernel op into matching models. It is compiled into the GPU plugin, not into
the megakernel library; shared helpers are in megakernelTransformationUtils.h.
The implementation is selected with -DMEGAKERNEL_IMPLEMENTATION=<folder name>.

The transformation fully defines the MegaKernel op (ov::intel_gpu::op::MegaKernelAttrs):
-> inputs: any number, in an order of the megakernel's choosing
-> outputs: any number, each with an element type and a shape whose dimensions are
   constants or copied from input dimensions (MegaKernelDim)
-> params: opaque bytes for model-specific attributes, usually a trivially copyable struct
   packed with mk::util::make_params and read back in Fill* with mk::ReadParams

The created library has to be shared lib with the same name as the main folder.

Each megakernel implementationt has to define:
class ConstantParamsImpl -> which inherits from mk::IConstantParams
class RuntimeParamsImpl -> which inherits from mk::IRuntimeParams
class PlatformParamsImpl -> which inherits from mk::IPlatformParams
