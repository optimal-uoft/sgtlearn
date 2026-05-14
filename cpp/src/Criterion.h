#pragma once

/**
 * @file Criterion.h
 * @brief Closed-form impurity and loss helpers shared by splitters and branch-assignment processors.
 */

#include <vector>

namespace Criterion {

/** Multiclass Shannon entropy from histogram counts. */
double entropy(const std::vector<size_t> &classCounts, size_t N);

/** Multiclass Gini impurity from histogram counts. */
double gini(const std::vector<size_t> &classCounts, size_t N);

/**
 * @param yPowerSum sum of y^(i + 1) accross all samples in set
 * @param N number of samples in set
 * @return MSE
 */
double squaredError(const std::vector<float> &yPowerSum, size_t N);

double gainAndHessian(const std::vector<float> &derivatives, double lambda);
} // namespace Criterion