/**
 * @file BranchAssignmentFactory.cpp
 * @brief Factory implementation for ``BranchAssignment`` objects.
 */

#include "BranchAssignmentFactory.h"

#include "AbsoluteErrorBranchAssignment.h"
#include "BranchAssignmentVariants.h"

#include <stdexcept>

std::unique_ptr<BranchAssignment> makeBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<double>> &leafStats,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts, size_t numClasses,
    std::vector<std::vector<float>> *maeLeafYs,
    std::vector<std::vector<float>> *maeLeafWs) {
  switch (criterion) {
  case LearningCriterion::Entropy:
    if (numClasses == 0)
      throw std::invalid_argument(
          "makeBranchAssignment(Entropy): numClasses must be positive");
    return std::make_unique<EntropyBranchAssignment>(
        assignments, numPartitions, leafStats, leafWeights, leafSampleCounts,
        numClasses);
  case LearningCriterion::Gini:
    if (numClasses == 0)
      throw std::invalid_argument(
          "makeBranchAssignment(Gini): numClasses must be positive");
    return std::make_unique<GiniBranchAssignment>(
        assignments, numPartitions, leafStats, leafWeights, leafSampleCounts,
        numClasses);
  case LearningCriterion::SquaredError:
    return std::make_unique<SquaredErrorBranchAssignment>(
        assignments, numPartitions, leafStats, leafWeights, leafSampleCounts);
  case LearningCriterion::AbsoluteError:
    if (!maeLeafYs || !maeLeafWs)
      throw std::invalid_argument(
          "makeBranchAssignment(AbsoluteError): maeLeafYs and maeLeafWs "
          "required");
    return std::make_unique<AbsoluteErrorBranchAssignment>(
        assignments, numPartitions, *maeLeafYs, *maeLeafWs, leafWeights,
        leafSampleCounts);
  default:
    throw std::invalid_argument(
        "makeBranchAssignment: unsupported learning criterion");
  }
}
