#pragma once

#include <vector>

#include <CL/sycl.hpp>

namespace jacobi {

struct CompResult {
    std::vector<float> x;
    double elapsed;
    int iter;
    float accuracy;
};

CompResult calculateWithAccessor(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                 float accuracyTarget, sycl::queue &queue);
CompResult calculateWithSharedMemory(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue);
CompResult calculateWithDeviceMemory(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue);

} // namespace jacobi
