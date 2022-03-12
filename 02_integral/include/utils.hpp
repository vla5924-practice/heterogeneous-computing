#pragma once

#include <string_view>

#include <CL/sycl.hpp>

namespace utils {

sycl::queue createDeviceQueueByType(std::string_view deviceType);

} // namespace utils
