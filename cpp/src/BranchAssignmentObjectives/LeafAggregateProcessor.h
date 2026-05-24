#pragma once

/**
 * @file LeafAggregateProcessor.h
 * @brief Strategy objects that map aggregated leaf statistics to a scalar partition loss (entropy, Gini, MSE, …).
 */

#include "Criterion.h"
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

class EntropyProcessor final : public ILeafAggregateProcessor<double> {
public:
  double compute(const std::vector<double> &aggregatedStats,
                 double totalWeight) const override {
    (void)totalWeight;
    double w = 0.0;
    for (double c : aggregatedStats)
      w += c;
    return Criterion::entropy(aggregatedStats, w);
  }
};

class GiniProcessor final : public ILeafAggregateProcessor<double> {
public:
  double compute(const std::vector<double> &aggregatedStats,
                 double totalWeight) const override {
    (void)totalWeight;
    double w = 0.0;
    for (double c : aggregatedStats)
      w += c;
    return Criterion::gini(aggregatedStats, w);
  }
};

class SquaredErrorProcessor final : public ILeafAggregateProcessor<float> {
public:
  double compute(const std::vector<float> &aggregatedStats,
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
