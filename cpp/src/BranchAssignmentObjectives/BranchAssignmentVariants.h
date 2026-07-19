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

namespace branch_assignment_detail {
inline size_t totalClasses(const std::vector<size_t> &classesPerOutput) {
  size_t total = 0;
  for (size_t nc : classesPerOutput)
    total += nc;
  return total;
}
} // namespace branch_assignment_detail

/** Classification: statsDim = sum(classesPerOutput); summed per-output entropy. */
class EntropyBranchAssignment
    : public leaf_aggregate::LeafAggregationBranchAssignment<double> {
public:
  EntropyBranchAssignment(std::vector<size_t> &assignments, size_t numPartitions,
                          std::vector<std::vector<double>> &stats,
                          std::vector<double> &leafWeights,
                          const std::vector<size_t> &leafSampleCounts,
                          const std::vector<size_t> &classesPerOutput)
      : leaf_aggregate::LeafAggregationBranchAssignment<double>(
            assignments, numPartitions, stats, leafWeights, leafSampleCounts,
            branch_assignment_detail::totalClasses(classesPerOutput),
            std::make_unique<leaf_aggregate::EntropyProcessor>(
                classesPerOutput)) {}
};

/** Classification: statsDim = sum(classesPerOutput); summed per-output Gini. */
class GiniBranchAssignment
    : public leaf_aggregate::LeafAggregationBranchAssignment<double> {
public:
  GiniBranchAssignment(std::vector<size_t> &assignments, size_t numPartitions,
                       std::vector<std::vector<double>> &stats,
                       std::vector<double> &leafWeights,
                       const std::vector<size_t> &leafSampleCounts,
                       const std::vector<size_t> &classesPerOutput)
      : leaf_aggregate::LeafAggregationBranchAssignment<double>(
            assignments, numPartitions, stats, leafWeights, leafSampleCounts,
            branch_assignment_detail::totalClasses(classesPerOutput),
            std::make_unique<leaf_aggregate::GiniProcessor>(classesPerOutput)) {}
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
            std::make_unique<leaf_aggregate::SquaredErrorProcessor>(nOutputs)) {}
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
