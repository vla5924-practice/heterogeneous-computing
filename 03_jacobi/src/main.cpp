#include <cstdlib>
#include <iostream>
#include <string_view>

#include <CL/sycl.hpp>

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

    auto [A, b] = utils::generateEquationSystem(rowsCount);

    std::vector<float> x0;
    std::vector<float> x1 = b;
    int iter = 0;
    float accuracy = 0;

    do {
        x0 = x1;

        try {
            sycl::buffer<float> aBuffer(A.data(), A.size());
            sycl::buffer<float> bBuffer(b.data(), b.size());

            size_t globalSize = x0.size();
            sycl::buffer<float> x0Buffer(x0.data(), globalSize);
            sycl::buffer<float> x1Buffer(x1.data(), globalSize);

            sycl::event event = queue.submit([&](sycl::handler &h) {
                auto aHandle = aBuffer.get_access<sycl::access::mode::read>(h);
                auto bHandle = bBuffer.get_access<sycl::access::mode::read>(h);

                auto x0Handle = x0Buffer.get_access<sycl::access::mode::read>(h);
                auto x1Handle = x1Buffer.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(globalSize), [=](sycl::item<1> item) {
                    int i = item.get_id(0);
                    int n = item.get_range(0);
                    float s = 0;
                    for (int j = 0; j < n; j++)
                        s += i != j ? aHandle[j * n + i] * x0Handle[j] : 0;
                    x1Handle[i] = (bHandle[i] - s) / aHandle[i * n + i];
                });
            });
            queue.wait();
        } catch (std::exception &e) {
            std::cout << e.what() << std::endl;
        }

        accuracy = utils::norm(x0, x1);
        iter++;
    } while (iter < iterationsLimit && accuracy > accuracyTarget);

    std::vector<float> computed = x1;
}
