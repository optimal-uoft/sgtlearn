#pragma once

/**
 * @file BranchAssignmentVariants.h
 * @brief Concrete ``BranchAssignment`` types for entropy, Gini, squared error, and gain/hessian objectives.
 */

#include <cstddef>
#include "AbsoluteErrorBranchAssignment.h"
#include "BranchAssignmentFactory.h"
#include "LeafAggregationBranchAssignment.h"
#include <memory>
#include <vector>

/** Classification: nested ``[leaf][output][class]``; statsDim = nOutputs. */
class EntropyBranchAssignment
    : public leaf_aggregate::LeafAggregationBranchAssignment<
          std::vector<double>> {
public:
  EntropyBranchAssignment(
      std::vector<size_t> &assignments, size_t numPartitions,
      std::vector<std::vector<std::vector<double>>> &stats,
      std::vector<double> &leafWeights,
      const std::vector<size_t> &leafSampleCounts,
      const std::vector<size_t> &classesPerOutput)
      : leaf_aggregate::LeafAggregationBranchAssignment<std::vector<double>>(
            assignments, numPartitions, stats, leafWeights, leafSampleCounts,
            classesPerOutput.size(),
            std::make_unique<leaf_aggregate::EntropyProcessor>()) {}
};

/** Classification: nested ``[leaf][output][class]``; statsDim = nOutputs. */
class GiniBranchAssignment
    : public leaf_aggregate::LeafAggregationBranchAssignment<
          std::vector<double>> {
public:
  GiniBranchAssignment(std::vector<size_t> &assignments, size_t numPartitions,
                       std::vector<std::vector<std::vector<double>>> &stats,
                       std::vector<double> &leafWeights,
                       const std::vector<size_t> &leafSampleCounts,
                       const std::vector<size_t> &classesPerOutput)
      : leaf_aggregate::LeafAggregationBranchAssignment<std::vector<double>>(
            assignments, numPartitions, stats, leafWeights, leafSampleCounts,
            classesPerOutput.size(),
            std::make_unique<leaf_aggregate::GiniProcessor>()) {}
};

/** Regression MSE: statsDim = 2 * nOutputs (per-output sum y, sum y²). */
class SquaredErrorBranchAssignment
    : public leaf_aggregate::LeafAggregationBranchAssignment<double> {
public:
  SquaredErrorBranchAssignment(std::vector<size_t> &assignments,
                               size_t numPartitions,
                               std::vector<std::vector<double>> &stats,
                               std::vector<double> &leafWeights,
                               const std::vector<size_t> &leafSampleCounts,
                               size_t nOutputs = 1)
      : leaf_aggregate::LeafAggregationBranchAssignment<double>(
            assignments, numPartitions, stats, leafWeights, leafSampleCounts,
            2 * nOutputs,
            std::make_unique<leaf_aggregate::SquaredErrorProcessor>()) {}
};

/** Gradient boosting: statsDim = 2 (sum g, sum h); L2 leaf regularization λ. */
class GainHessianBranchAssignment
    : public leaf_aggregate::LeafAggregationBranchAssignment<float> {
public:
  static constexpr size_t kStatsDim = 2;

  GainHessianBranchAssignment(std::vector<size_t> &assignments,
                              size_t numPartitions,
                              std::vector<std::vector<float>> &stats,
                              std::vector<double> &leafWeights,
                              const std::vector<size_t> &leafSampleCounts,
                              double lambda)
      : leaf_aggregate::LeafAggregationBranchAssignment<float>(
            assignments, numPartitions, stats, leafWeights, leafSampleCounts,
            kStatsDim,
            std::make_unique<leaf_aggregate::GainHessianProcessor>(lambda)) {}
};
