/**
 * @file BranchAssignmentFactory.cpp
 * @brief Factory implementation for ``BranchAssignment`` objects.
 */

#include <memory>
#include <cstddef>
#include "BranchAssignmentFactory.h"

#include "AbsoluteErrorBranchAssignment.h"
#include "BranchAssignmentVariants.h"

#include <stdexcept>

std::unique_ptr<BranchAssignment> makeBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<double>> &leafStats,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts,
    const std::vector<size_t> &classesPerOutput, size_t nOutputs,
    std::vector<std::vector<std::vector<float>>> *maeLeafYs,
    std::vector<std::vector<float>> *maeLeafWs) {
  (void)classesPerOutput;
  switch (criterion) {
  case LearningCriterion::SquaredError:
    return std::make_unique<SquaredErrorBranchAssignment>(
        assignments, numPartitions, leafStats, leafWeights, leafSampleCounts,
        nOutputs);
  case LearningCriterion::AbsoluteError:
    if (!maeLeafYs || !maeLeafWs)
      throw std::invalid_argument(
          "makeBranchAssignment(AbsoluteError): maeLeafYs and maeLeafWs "
          "required");
    return std::make_unique<AbsoluteErrorBranchAssignment>(
        assignments, numPartitions, *maeLeafYs, *maeLeafWs, leafWeights,
        leafSampleCounts);
  case LearningCriterion::Entropy:
  case LearningCriterion::Gini:
    throw std::invalid_argument(
        "makeBranchAssignment: Entropy/Gini require nested class leaf stats "
        "(use the nested-stats overload)");
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
    const std::vector<size_t> &classesPerOutput) {
  switch (criterion) {
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
        "makeBranchAssignment: nested leaf stats overload supports Entropy/"
        "Gini only");
  }
}
