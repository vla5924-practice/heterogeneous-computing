#include <cstdlib>
#include <iostream>
#include <string_view>

#include <CL/sycl.hpp>
#include <omp.h>

#include "jacobi.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout << "Wrong arguments passed. Exiting." << std::endl;
        return 1;
    }

    size_t rowsCount = static_cast<size_t>(std::atoi(argv[1]));
    float accuracyTarget = std::atof(argv[2]);
    int iterationsLimit = std::atoi(argv[3]);
    std::string_view deviceType = argv[4];

    sycl::queue queue = utils::createDeviceQueueByType(deviceType);
    std::cout << "Target device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    auto [A, b] = utils::generateEquationSystem(rowsCount);

    {
        auto result = jacobi::calculateWithAccessor(A, b, iterationsLimit, accuracyTarget, queue);
        float deviation = utils::deviation(A, b, result.x);
        std::cout << "[Accessor]  Time: " << result.elapsed << " ms  Deviation: " << deviation << std::endl;
    }
    {
        auto result = jacobi::calculateWithSharedMemory(A, b, iterationsLimit, accuracyTarget, queue);
        float deviation = utils::deviation(A, b, result.x);
        std::cout << "[Shared]    Time: " << result.elapsed << " ms  Deviation: " << deviation << std::endl;
    }
    {
        auto result = jacobi::calculateWithDeviceMemory(A, b, iterationsLimit, accuracyTarget, queue);
        float deviation = utils::deviation(A, b, result.x);
        std::cout << "[Device]    Time: " << result.elapsed << " ms  Deviation: " << deviation << std::endl;
    }
}
