#pragma once

#include <string_view>
#include <vector>

#include <CL/sycl.hpp>

namespace utils {

sycl::queue createDeviceQueueByType(std::string_view deviceType);
std::pair<std::vector<float>, std::vector<float>> generateEquationSystem(size_t rowsCount);
float norm(const std::vector<float> &x0, const std::vector<float> &x1);
float deviation(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& x);

} // namespace utils
