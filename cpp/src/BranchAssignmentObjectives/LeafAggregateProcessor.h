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
 * Multi-output entropy: ``aggregatedStats`` is the concatenation of one
 * histogram per output; returns the SUM of per-output entropy. A single-entry
 * ``classesPerOutput`` reproduces the scalar path.
 */
class EntropyProcessor final : public ILeafAggregateProcessor<double> {
public:
  explicit EntropyProcessor(std::vector<size_t> classesPerOutput)
      : classesPerOutput_(std::move(classesPerOutput)) {}

  double compute(const std::vector<double> &aggregatedStats,
                 double totalWeight) const override {
    (void)totalWeight;
    return Criterion::entropyMulti(aggregatedStats, classesPerOutput_);
  }

private:
  std::vector<size_t> classesPerOutput_;
};

/** Multi-output Gini: SUM of per-output Gini over concatenated histograms. */
class GiniProcessor final : public ILeafAggregateProcessor<double> {
public:
  explicit GiniProcessor(std::vector<size_t> classesPerOutput)
      : classesPerOutput_(std::move(classesPerOutput)) {}

  double compute(const std::vector<double> &aggregatedStats,
                 double totalWeight) const override {
    (void)totalWeight;
    return Criterion::giniMulti(aggregatedStats, classesPerOutput_);
  }

private:
  std::vector<size_t> classesPerOutput_;
};

/**
 * Multi-output MSE: ``aggregatedStats`` is ``[Σw·y0, Σw·y0², ...]`` (length
 * ``2 * nOutputs``); returns the SUM of per-output squared error.
 */
class SquaredErrorProcessor final : public ILeafAggregateProcessor<double> {
public:
  explicit SquaredErrorProcessor(size_t nOutputs = 1) : nOutputs_(nOutputs) {}

  double compute(const std::vector<double> &aggregatedStats,
                 double totalWeight) const override {
    return Criterion::squaredErrorMulti(aggregatedStats, totalWeight, nOutputs_);
  }

private:
  size_t nOutputs_;
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
