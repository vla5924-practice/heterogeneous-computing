#include "utils.hpp"

#include <random>

namespace {

static std::mt19937 &mersenneInstance() {
    static std::random_device rd;
    static std::mt19937 mersenne(rd());
    return mersenne;
}

static void fillRandomly(std::vector<float> &arr) {
    std::uniform_real_distribution<> urd(1.0, 3.0);
    for (float &el : arr)
        el = urd(mersenneInstance());
}

static float vectorLength(const float *x, size_t n) {
    float s = 0;
    for (size_t i = 0; i < n; i++) {
        s += x[i] * x[i];
    }
    return std::sqrt(s);
}

static float normAbs(const float *x0, const float *x1, size_t n) {
    float s = 0;
    for (size_t i = 0; i < n; i++) {
        s += (x0[i] - x1[i]) * (x0[i] - x1[i]);
    }
    return std::sqrt(s);
}

static float normRel(const float *x0, const float *x1, size_t n) {
    return normAbs(x0, x1, n) / vectorLength(x0, n);
}

static float deviationAbs(const float *a, const float *b, const float *x, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) {
        float s = 0;
        for (int j = 0; j < n; j++) {
            s += a[j * n + i] * x[j];
        }
        s -= b[i];
        norm += s * s;
    }
    return sqrt(norm);
}

static float deviationRel(const float *a, const float *b, const float *x, int n) {
    return deviationAbs(a, b, x, n) / vectorLength(b, n);
}

} // namespace

namespace utils {

sycl::queue createDeviceQueueByType(std::string_view deviceType) {
    sycl::property_list props{sycl::property::queue::enable_profiling()};
    if (deviceType == "cpu")
        return {sycl::cpu_selector{}, props};
    if (deviceType == "gpu")
        return {sycl::gpu_selector{}, props};
    return {sycl::default_selector{}, props};
}

std::pair<std::vector<float>, std::vector<float>> generateEquationSystem(size_t rowsCount) {
    std::vector<float> matrix(rowsCount * rowsCount, 0.f);
    fillRandomly(matrix);
    std::uniform_real_distribution<> urd(rowsCount * 5.0, rowsCount * 5.0 + 2.0);
    for (size_t i = 0; i < rowsCount; i++)
        matrix[i * rowsCount + i] = urd(mersenneInstance());
    std::vector<float> col(rowsCount, 0.f);
    fillRandomly(col);
    return {matrix, col};
}

float norm(const float *x0, const float *x1, size_t n) {
    return normRel(x0, x1, n);
}

float norm(const std::vector<float> &x0, const std::vector<float> &x1) {
    return normRel(x0.data(), x1.data(), x0.size());
}

float deviation(const std::vector<float> &a, const std::vector<float> &b, const std::vector<float> &x) {
    size_t n = x.size();
    return deviationRel(a.data(), b.data(), x.data(), n);
}

} // namespace utils
