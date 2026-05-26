#pragma once

/**
 * @file Criterion.h
 * @brief Closed-form impurity and loss helpers shared by splitters and branch-assignment processors.
 */

#include <vector>

namespace Criterion {

/** Multiclass Shannon entropy from weighted histogram counts. */
double entropy(const std::vector<double> &classCounts, double totalWeight);

/** Multiclass Gini impurity from weighted histogram counts. */
double gini(const std::vector<double> &classCounts, double totalWeight);

/**
 * @param yPowerSum weighted sum of y and y^2 (index 0 and 1).
 * @param totalWeight sum of sample weights in the set.
 * @return weighted MSE
 */
double squaredError(const std::vector<double> &yPowerSum, double totalWeight);

/** Weighted median and mean MAE (sklearn ``precompute_absolute_errors``). */
struct AbsoluteErrorStats {
  double median = 0.0;
  double mae = 0.0;
  double totalWeight = 0.0;
};

AbsoluteErrorStats absoluteError(const std::vector<float> &ys,
                                 const std::vector<float> &weights);

double gainAndHessian(const std::vector<float> &derivatives, double lambda);
} // namespace Criterion