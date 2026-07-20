#pragma once

/**
 * @file LeafAggregateProcessor.h
 * @brief Strategy objects that map aggregated leaf statistics to a scalar partition loss (entropy, Gini, MSE, …).
 */

#include "Criterion.h"
#include <cstddef>
#include <utility>
#include <vector>

namespace leaf_aggregate {

/** Computes partition impurity / loss from aggregated per-leaf statistics. */
template <typename T>
class ILeafAggregateProcessor {
public:
  virtual ~ILeafAggregateProcessor() = default;

  virtual double compute(const std::vector<T> &aggregatedStats,
                         double totalWeight) const = 0;
};

/**
 * Multi-output entropy: ``aggregatedStats[o]`` is the histogram for output
 * ``o``; returns the SUM of per-output entropy.
 */
class EntropyProcessor final
    : public ILeafAggregateProcessor<std::vector<double>> {
public:
  double compute(const std::vector<std::vector<double>> &aggregatedStats,
                 double totalWeight) const override {
    (void)totalWeight;
    return Criterion::entropy(aggregatedStats);
  }
};

/** Multi-output Gini: SUM of per-output Gini over nested histograms. */
class GiniProcessor final
    : public ILeafAggregateProcessor<std::vector<double>> {
public:
  double compute(const std::vector<std::vector<double>> &aggregatedStats,
                 double totalWeight) const override {
    (void)totalWeight;
    return Criterion::gini(aggregatedStats);
  }
};

/**
 * Multi-output MSE: ``aggregatedStats`` is ``[Σw·y0, Σw·y0², ...]`` (length
 * ``2 * nOutputs``); returns the SUM of per-output squared error.
 */
class SquaredErrorProcessor final : public ILeafAggregateProcessor<double> {
public:
  double compute(const std::vector<double> &aggregatedStats,
                 double totalWeight) const override {
    return Criterion::squaredError(aggregatedStats, totalWeight);
  }
};

class GainHessianProcessor final : public ILeafAggregateProcessor<float> {
public:
  explicit GainHessianProcessor(double lambda) : lambda_(lambda) {}

  double compute(const std::vector<float> &aggregatedStats,
                 double totalWeight) const override {
    (void)totalWeight;
    return Criterion::gainAndHessian(aggregatedStats, lambda_);
  }

private:
  double lambda_;
};

} // namespace leaf_aggregate
