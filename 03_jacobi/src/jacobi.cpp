#include "jacobi.hpp"

#include <omp.h>

#include "utils.hpp"

namespace jacobi {

CompResult calculateWithAccessor(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                 float accuracyTarget, sycl::queue &queue) {
    CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    std::vector<float> x0;
    std::vector<float> x1 = b;

    sycl::buffer<float> aBuffer(A.data(), A.size());
    sycl::buffer<float> bBuffer(b.data(), b.size());
    size_t globalSize = b.size();

    double begin = omp_get_wtime();
    do {
        x0 = x1;

        {
            sycl::buffer<float> x0Buffer(x0.data(), globalSize);
            sycl::buffer<float> x1Buffer(x1.data(), globalSize);

            sycl::event event = queue.submit([&](sycl::handler &h) {
                auto aHandle = aBuffer.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(h);
                auto bHandle = bBuffer.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(h);

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
        }

        result.accuracy = utils::norm(x0, x1);
        result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
    double end = omp_get_wtime();

    result.elapsed = (end - begin) * 1000.;
    result.x = x1;

    return result;
}

CompResult calculateWithSharedMemory(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue) {
    CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    size_t globalSize = b.size();
    size_t bSize = globalSize * sizeof(float);

    float *aShared = sycl::malloc_shared<float>(A.size(), queue);
    float *bShared = sycl::malloc_shared<float>(b.size(), queue);
    float *x0Shared = sycl::malloc_shared<float>(b.size(), queue);
    float *x1Shared = sycl::malloc_shared<float>(b.size(), queue);

    queue.memcpy(aShared, A.data(), A.size() * sizeof(float)).wait();
    queue.memcpy(bShared, b.data(), bSize).wait();
    queue.memcpy(x1Shared, b.data(), bSize).wait();

    std::vector<float> x0(globalSize);
    std::vector<float> x1(globalSize);

    double begin = omp_get_wtime();
    do {
        queue.memcpy(x0Shared, x1Shared, bSize).wait();
        sycl::event event = queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(globalSize), [=](sycl::item<1> item) {
                int i = item.get_id(0);
                int n = item.get_range(0);
                float s = 0;
                for (int j = 0; j < n; j++)
                    s += i != j ? aShared[j * n + i] * x0Shared[j] : 0;
                x1Shared[i] = (bShared[i] - s) / aShared[i * n + i];
            });
        });
        queue.wait();
        queue.memcpy(x0.data(), x0Shared, bSize).wait();
        queue.memcpy(x1.data(), x1Shared, bSize).wait();

        result.accuracy = utils::norm(x0, x1);
        result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
    double end = omp_get_wtime();

    sycl::free(aShared, queue);
    sycl::free(bShared, queue);
    sycl::free(x0Shared, queue);
    sycl::free(x1Shared, queue);

    result.elapsed = (end - begin) * 1000.;
    result.x = x1;

    return result;
}

CompResult calculateWithDeviceMemory(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue) {
    CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    size_t globalSize = b.size();
    size_t bSize = globalSize * sizeof(float);

    float *aDevice = sycl::malloc_device<float>(A.size(), queue);
    float *bDevice = sycl::malloc_device<float>(b.size(), queue);
    float *x0Device = sycl::malloc_device<float>(b.size(), queue);
    float *x1Device = sycl::malloc_device<float>(b.size(), queue);

    queue.memcpy(aDevice, A.data(), A.size() * sizeof(float)).wait();
    queue.memcpy(bDevice, b.data(), bSize).wait();
    queue.memcpy(x1Device, b.data(), bSize).wait();

    std::vector<float> x0(globalSize);
    std::vector<float> x1(globalSize);

    double begin = omp_get_wtime();
    do {
        queue.memcpy(x0Device, x1Device, bSize).wait();
        sycl::event event = queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(globalSize), [=](sycl::item<1> item) {
                int i = item.get_id(0);
                int n = item.get_range(0);
                float s = 0;
                for (int j = 0; j < n; j++)
                    s += i != j ? aDevice[j * n + i] * x1Device[j] : 0;
                x1Device[i] = (bDevice[i] - s) / aDevice[i * n + i];
            });
        });
        queue.wait();
        queue.memcpy(x0.data(), x0Device, bSize).wait();
        queue.memcpy(x1.data(), x1Device, bSize).wait();

        result.accuracy = utils::norm(x0, x1);
        result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
    double end = omp_get_wtime();

    sycl::free(aDevice, queue);
    sycl::free(bDevice, queue);
    sycl::free(x0Device, queue);
    sycl::free(x1Device, queue);

    result.elapsed = (end - begin) * 1000.;
    result.x = x1;

    return result;
}

} // namespace jacobi
