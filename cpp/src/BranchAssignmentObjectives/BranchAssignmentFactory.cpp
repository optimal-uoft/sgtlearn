/**
 * @file BranchAssignmentFactory.cpp
 * @brief Factory implementation for ``BranchAssignment`` objects.
 */

#include "BranchAssignmentFactory.h"

#include "AbsoluteErrorBranchAssignment.h"
#include "LeafAggregationBranchAssignment.h"

#include <cstddef>
#include <memory>
#include <stdexcept>

std::unique_ptr<BranchAssignment> makeBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts,
    std::vector<std::vector<std::vector<float>>> *maeLeafYs,
    std::vector<std::vector<float>> *maeLeafWs) {
  if (criterion != LearningCriterion::AbsoluteError)
    throw std::invalid_argument(
        "makeBranchAssignment: raw-sample overload supports AbsoluteError only");
  if (!maeLeafYs || !maeLeafWs)
    throw std::invalid_argument(
        "makeBranchAssignment(AbsoluteError): maeLeafYs and maeLeafWs required");
  return std::make_unique<AbsoluteErrorBranchAssignment>(
      assignments, numPartitions, *maeLeafYs, *maeLeafWs, leafWeights,
      leafSampleCounts);
}

std::unique_ptr<BranchAssignment> makeBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions,
    std::vector<std::vector<std::vector<double>>> &leafStats,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts,
    const std::vector<size_t> &classesPerOutput, size_t nOutputs) {
  switch (criterion) {
  case LearningCriterion::SquaredError:
    return std::make_unique<LeafAggregationBranchAssignment>(
        criterion, assignments, numPartitions, leafStats, leafWeights,
        leafSampleCounts, nOutputs);
  case LearningCriterion::Entropy:
  case LearningCriterion::Gini:
    if (classesPerOutput.empty())
      throw std::invalid_argument(
          "makeBranchAssignment(classification): classesPerOutput must be non-empty");
    return std::make_unique<LeafAggregationBranchAssignment>(
        criterion, assignments, numPartitions, leafStats, leafWeights,
        leafSampleCounts, classesPerOutput.size());
  default:
    throw std::invalid_argument(
        "makeBranchAssignment: nested leaf stats overload supports "
        "SquaredError/Entropy/Gini only");
  }
}
