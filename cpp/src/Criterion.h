#pragma once
#include <vector>

namespace Criterion {

double entropy(const std::vector<size_t> &classCounts, size_t N);

double gini(const std::vector<size_t> &classCounts, size_t N);

/**
 * @param yPowerSum sum of y^(i + 1) accross all samples in set
 * @param N number of samples in set
 * @return MSE
 */
double squaredError(const std::vector<float> &yPowerSum, size_t N);

} // namespace Criterion