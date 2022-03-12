#include "utils.hpp"

sycl::queue utils::createDeviceQueueByType(std::string_view deviceType) {
    sycl::property_list props{sycl::property::queue::enable_profiling()};
    if (deviceType == "cpu")
        return {sycl::cpu_selector{}, props};
    if (deviceType == "gpu")
        return {sycl::gpu_selector{}, props};
    return {sycl::default_selector{}, props};
}
