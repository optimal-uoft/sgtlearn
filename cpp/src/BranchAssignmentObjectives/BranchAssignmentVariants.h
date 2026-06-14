#pragma once

/**
 * @file BranchAssignmentVariants.h
 * @brief Concrete ``BranchAssignment`` types for entropy, Gini, squared error, and gain/hessian objectives.
 */

#include "AbsoluteErrorBranchAssignment.h"
#include "BranchAssignmentFactory.h"
#include "LeafAggregationBranchAssignment.h"
#include <memory>
#include <vector>

/** Classification: statsDim = numClasses; entropy impurity. */
class EntropyBranchAssignment
    : public leaf_aggregate::LeafAggregationBranchAssignment<double> {
public:
  EntropyBranchAssignment(std::vector<size_t> &assignments, size_t numPartitions,
                          std::vector<std::vector<double>> &stats,
                          std::vector<double> &leafWeights,
                          const std::vector<size_t> &leafSampleCounts,
                          size_t numClasses)
      : leaf_aggregate::LeafAggregationBranchAssignment<double>(
            assignments, numPartitions, stats, leafWeights, leafSampleCounts,
            numClasses, std::make_unique<leaf_aggregate::EntropyProcessor>()) {}
};

/** Classification: statsDim = numClasses; Gini impurity. */
class GiniBranchAssignment
    : public leaf_aggregate::LeafAggregationBranchAssignment<double> {
public:
  GiniBranchAssignment(std::vector<size_t> &assignments, size_t numPartitions,
                       std::vector<std::vector<double>> &stats,
                       std::vector<double> &leafWeights,
                       const std::vector<size_t> &leafSampleCounts,
                       size_t numClasses)
      : leaf_aggregate::LeafAggregationBranchAssignment<double>(
            assignments, numPartitions, stats, leafWeights, leafSampleCounts,
            numClasses, std::make_unique<leaf_aggregate::GiniProcessor>()) {}
};

/** Regression MSE: statsDim = 2 (sum y, sum y²). */
class SquaredErrorBranchAssignment
    : public leaf_aggregate::LeafAggregationBranchAssignment<double> {
public:
  static constexpr size_t kStatsDim = 2;

  SquaredErrorBranchAssignment(std::vector<size_t> &assignments,
                               size_t numPartitions,
                               std::vector<std::vector<double>> &stats,
                               std::vector<double> &leafWeights,
                               const std::vector<size_t> &leafSampleCounts)
      : leaf_aggregate::LeafAggregationBranchAssignment<double>(
            assignments, numPartitions, stats, leafWeights, leafSampleCounts,
            kStatsDim,
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
