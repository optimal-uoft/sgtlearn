#pragma once

#include "Criterion.h"
#include <vector>

namespace leaf_aggregate {

/** Computes partition impurity / loss from aggregated per-leaf statistics. */
template <typename T>
class ILeafAggregateProcessor {
public:
  virtual ~ILeafAggregateProcessor() = default;

  virtual double compute(const std::vector<T> &aggregatedStats,
                         size_t numSamples) const = 0;
};

class EntropyProcessor final : public ILeafAggregateProcessor<size_t> {
public:
  double compute(const std::vector<size_t> &aggregatedStats,
                 size_t numSamples) const override {
    return Criterion::entropy(aggregatedStats, numSamples);
  }
};

class GiniProcessor final : public ILeafAggregateProcessor<size_t> {
public:
  double compute(const std::vector<size_t> &aggregatedStats,
                 size_t numSamples) const override {
    return Criterion::gini(aggregatedStats, numSamples);
  }
};

class SquaredErrorProcessor final : public ILeafAggregateProcessor<float> {
public:
  double compute(const std::vector<float> &aggregatedStats,
                 size_t numSamples) const override {
    return Criterion::squaredError(aggregatedStats, numSamples);
  }
};

class GainHessianProcessor final : public ILeafAggregateProcessor<float> {
public:
  explicit GainHessianProcessor(double lambda) : lambda_(lambda) {}

  double compute(const std::vector<float> &aggregatedStats,
                 size_t numSamples) const override {
    (void)numSamples;
    return Criterion::gainAndHessian(aggregatedStats, lambda_);
  }

private:
  double lambda_;
};

} // namespace leaf_aggregate
