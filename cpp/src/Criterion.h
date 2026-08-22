#pragma once

/**
 * @file Criterion.h
 * @brief Closed-form impurity and loss helpers shared by splitters and branch-assignment processors.
 */

#include <cstddef>
#include <vector>

namespace Criterion {

/**
 * Shannon entropy over per-output class histograms.
 * ``countsByOutput[o]`` is the weighted class histogram for output ``o``.
 * Returns the SUM of per-output entropy (each row normalizes by its own
 * weight sum). Single-output is a length-1 outer vector.
 */
double entropy(const std::vector<std::vector<double>> &countsByOutput);

/**
 * Gini impurity over per-output class histograms.
 * Same layout and aggregation as ``entropy``.
 */
double gini(const std::vector<std::vector<double>> &countsByOutput);

/**
 * Weighted MSE. ``momentsByOutput[o]`` is ``[Σw·y, Σw·y²]`` for output ``o``.
 * Returns the SUM of per-output MSE. Single-output is a length-1 outer vector.
 */
double squaredError(const std::vector<std::vector<double>> &momentsByOutput,
                    double totalWeight);

/** Weighted median and mean MAE (sklearn ``precompute_absolute_errors``). */
struct AbsoluteErrorStats {
  double median = 0.0;
  double mae = 0.0;
  double totalWeight = 0.0;
};

AbsoluteErrorStats absoluteError(const std::vector<float> &ys,
                                 const std::vector<float> &weights);

/**
 * Same as ``absoluteError`` but assumes ``ys`` are already sorted ascending
 * (weights aligned). Skips the internal sort — used by merge/filter MAE CD.
 */
AbsoluteErrorStats absoluteErrorPresorted(const std::vector<float> &ys,
                                          const std::vector<float> &weights);

double gainAndHessian(const std::vector<float> &derivatives, double lambda);
} // namespace Criterion
