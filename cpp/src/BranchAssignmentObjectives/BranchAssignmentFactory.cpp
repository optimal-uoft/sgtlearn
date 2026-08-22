/**
 * @file BranchAssignmentFactory.cpp
 * @brief Factory implementation for ``BranchAssignment`` objects.
 */

#include <memory>
#include <cstddef>
#include "BranchAssignmentFactory.h"

#include "AbsoluteErrorBranchAssignment.h"
#include "BranchAssignmentVariants.h"
#include "MaeBranchConfig.h"

#include <stdexcept>

std::unique_ptr<BranchAssignment> makeBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<double>> &leafStats,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts,
    std::vector<std::vector<std::vector<float>>> *maeLeafYs,
    std::vector<std::vector<float>> *maeLeafWs) {
  (void)leafStats;
  switch (criterion) {
  case LearningCriterion::AbsoluteError:
    if (!maeLeafYs || !maeLeafWs)
      throw std::invalid_argument(
          "makeBranchAssignment(AbsoluteError): maeLeafYs and maeLeafWs "
          "required");
    if (mae_branch_config::backend() == mae_branch_config::Backend::Sort)
      return std::make_unique<AbsoluteErrorBranchAssignmentSort>(
          assignments, numPartitions, *maeLeafYs, *maeLeafWs, leafWeights,
          leafSampleCounts);
    return std::make_unique<AbsoluteErrorBranchAssignment>(
        assignments, numPartitions, *maeLeafYs, *maeLeafWs, leafWeights,
        leafSampleCounts);
  case LearningCriterion::SquaredError:
  case LearningCriterion::Entropy:
  case LearningCriterion::Gini:
    throw std::invalid_argument(
        "makeBranchAssignment: SquaredError/Entropy/Gini require nested leaf "
        "stats (use the nested-stats overload)");
  default:
    throw std::invalid_argument(
        "makeBranchAssignment: unsupported learning criterion");
  }
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
    return std::make_unique<SquaredErrorBranchAssignment>(
        assignments, numPartitions, leafStats, leafWeights, leafSampleCounts,
        nOutputs);
  case LearningCriterion::Entropy:
    if (classesPerOutput.empty())
      throw std::invalid_argument(
          "makeBranchAssignment(Entropy): classesPerOutput must be non-empty");
    return std::make_unique<EntropyBranchAssignment>(
        assignments, numPartitions, leafStats, leafWeights, leafSampleCounts,
        classesPerOutput);
  case LearningCriterion::Gini:
    if (classesPerOutput.empty())
      throw std::invalid_argument(
          "makeBranchAssignment(Gini): classesPerOutput must be non-empty");
    return std::make_unique<GiniBranchAssignment>(
        assignments, numPartitions, leafStats, leafWeights, leafSampleCounts,
        classesPerOutput);
  default:
    throw std::invalid_argument(
        "makeBranchAssignment: nested leaf stats overload supports "
        "SquaredError/Entropy/Gini only");
  }
}
