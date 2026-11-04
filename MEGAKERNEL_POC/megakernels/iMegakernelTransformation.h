#pragma once

#include <memory>

namespace ov {
class Model;
}

namespace mk {
bool InsertMegakernelTransformation(const std::shared_ptr<ov::Model>& model);

}  // namespace mk
