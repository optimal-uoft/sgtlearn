#pragma once

/**
 * @file Criterion.h
 * @brief Closed-form impurity and loss helpers shared by splitters and branch-assignment processors.
 */

#include <cstddef>
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

/**
 * Multi-output entropy: ``classCounts`` is the concatenation of one weighted
 * histogram per output (block ``o`` has ``classesPerOutput[o]`` entries).
 * Returns the SUM of per-output entropy. Each block's own summed weight is used
 * as its normalizer. For a single output this equals ``entropy``.
 */
double entropyMulti(const std::vector<double> &classCounts,
                    const std::vector<size_t> &classesPerOutput);

/** Multi-output Gini: SUM of per-output Gini over concatenated histograms. */
double giniMulti(const std::vector<double> &classCounts,
                 const std::vector<size_t> &classesPerOutput);

/**
 * Multi-output MSE: ``yPowerSum`` is ``[Σw·y0, Σw·y0², Σw·y1, Σw·y1², ...]``
 * (length ``2 * nOutputs``). Returns the SUM of per-output ``squaredError`` on
 * each ``(Σw·y, Σw·y²)`` pair. For a single output this equals ``squaredError``.
 */
double squaredErrorMulti(const std::vector<double> &yPowerSum,
                         double totalWeight, size_t nOutputs);

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