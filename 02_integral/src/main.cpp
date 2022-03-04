#include <iostream>
#include <string>

#include <CL/sycl.hpp>

sycl::queue createDeviceQueueByType(const std::string &deviceType) {
    sycl::property_list props{sycl::property::queue::enable_profiling()};
    if (deviceType == "cpu")
        return {sycl::cpu_selector{}, props};
    if (deviceType == "gpu")
        return {sycl::gpu_selector{}, props};
    return {sycl::default_selector{}, props};
}

int main(int argc, char *argv[]) {
    constexpr float expected = 0.3868223;
    int num_intervals = std::atoi(argv[1]);
    std::string device_type = argv[2];

    constexpr size_t GROUP_SIZE = 16;
    constexpr float a = 0;
    constexpr float b = 1;
    constexpr float c = 0;
    constexpr float d = 1;

    std::cout << "Number of rectangles: " << num_intervals << " x " << num_intervals << std::endl;

    float delta_x = (b - a) / num_intervals;
    float delta_y = (d - c) / num_intervals;

    sycl::queue queue = createDeviceQueueByType(device_type);
    std::cout << "Target device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    size_t num_groups = num_intervals / GROUP_SIZE + 1;
    std::vector<float> result(num_groups * num_groups, 0);
    sycl::buffer<float> buf_result(result.data(), result.size());

    try {
        sycl::event event = queue.submit([&](sycl::handler &h) {
            auto out_result = buf_result.get_access<sycl::access::mode::write>(h);
            h.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(num_intervals, num_intervals), sycl::range<2>(GROUP_SIZE, GROUP_SIZE)),
                [=](sycl::nd_item<2> item) {
                    float x = delta_x * (item.get_global_id(0) + 0.5);
                    float y = delta_y * (item.get_global_id(1) + 0.5);
                    float value = sycl::sin(x) * sycl::cos(y);
                    float sum = sycl::reduce_over_group(item.get_group(), value, std::plus<float>());
                    if (item.get_local_id(0) == 0 && item.get_local_id(1) == 0) {
                        out_result[item.get_group(0) * item.get_group_range(0) + item.get_group(1)] = sum;
                    }
                });
        });
        queue.wait();
        uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();

        float computed = 0;
        for (const auto &res : result) {
            computed += res;
        }
        computed *= delta_x * delta_y;

        std::cout << "Kernel Execution Time: " << (end - start) / 1e+6 << " ms" << std::endl;
        std::cout << "Expected: " << expected << std::endl;
        std::cout << "Computed: " << computed << std::endl;
        std::cout << "Difference: " << std::abs(computed - expected) << std::endl;

    } catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
