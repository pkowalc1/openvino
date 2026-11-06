#pragma once

#include <memory>

#include "iMegakernelRuntime.h"

namespace ov {
class Model;
}

namespace cldnn {
class primitive_inst;
}

namespace mk {
bool InsertMegakernelTransformation(const std::shared_ptr<ov::Model>& model);

std::shared_ptr<const mk::IConstantParams> GetMegaKernelConstantParams(cldnn::primitive_inst* prim);
std::shared_ptr<const mk::IRuntimeParams> GetMegaKernelRuntimeParams(cldnn::primitive_inst* prim);
}  // namespace mk
