#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string_view>

#include <CL/sycl.hpp>

sycl::queue createDeviceQueueByType(std::string_view deviceType) {
    sycl::property_list props{sycl::property::queue::enable_profiling()};
    if (deviceType == "cpu")
        return {sycl::cpu_selector{}, props};
    if (deviceType == "gpu")
        return {sycl::gpu_selector{}, props};
    return {sycl::default_selector{}, props};
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Wrong arguments passed. Exiting." << std::endl;
        return 1;
    }

    int stepsCount = std::atoi(argv[1]);
    std::string_view deviceType = argv[2];

    constexpr float expected = 0.3868223;
    constexpr size_t groupSize = 16;
    float dx = 1.f / stepsCount;
    float dy = 1.f / stepsCount;

    size_t groupsCount = stepsCount / groupSize + 1;
    std::vector<float> result(groupsCount * groupsCount, 0);
    sycl::buffer<float> resultBuffer(result.data(), result.size());

    sycl::queue queue = createDeviceQueueByType(deviceType);

    std::cout << "Number of rectangles: " << stepsCount << " x " << stepsCount << std::endl;
    std::cout << "Target device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    uint64_t start = 0;
    uint64_t end = 0;

    try {
        sycl::event event = queue.submit([&](sycl::handler &h) {
            auto buffer = resultBuffer.get_access<sycl::access::mode::write>(h);
            h.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(stepsCount, stepsCount), sycl::range<2>(groupSize, groupSize)),
                [=](sycl::nd_item<2> item) {
                    float x = dx * (item.get_global_id(0) + 0.5);
                    float y = dy * (item.get_global_id(1) + 0.5);
                    float value = sycl::sin(x) * sycl::cos(y);
                    float sum = sycl::reduce_over_group(item.get_group(), value, std::plus<float>());
                    if (item.get_local_id(0) == 0 && item.get_local_id(1) == 0) {
                        buffer[item.get_group(0) * item.get_group_range(0) + item.get_group(1)] = sum;
                    }
                });
        });
        queue.wait();

        start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    float computed = std::accumulate(result.begin(), result.end(), 0.f) * dx * dy;

    std::cout << "Kernel time: " << (end - start) / 1e+6 << " ms" << std::endl;
    std::cout << "Expected value: " << expected << std::endl;
    std::cout << "Computed value: " << computed << std::endl;
    std::cout << "Difference: " << std::abs(computed - expected) << std::endl;
}
