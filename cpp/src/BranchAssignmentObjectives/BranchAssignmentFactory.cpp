/**
 * @file BranchAssignmentFactory.cpp
 * @brief Factory implementations for classification and regression ``BranchAssignment`` objects.
 */

#include "BranchAssignmentFactory.h"

#include "AbsoluteErrorBranchAssignment.h"
#include "BranchAssignmentVariants.h"

#include <stdexcept>

std::unique_ptr<BranchAssignment> makeClassificationBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<double>> &classLeafStats,
    std::vector<double> &leafWeights, size_t numClasses) {
  if (numClasses == 0)
    throw std::invalid_argument(
        "makeClassificationBranchAssignment: numClasses must be positive");

  switch (criterion) {
  case LearningCriterion::Entropy:
    return std::make_unique<EntropyBranchAssignment>(
        assignments, numPartitions, classLeafStats, leafWeights, numClasses);
  case LearningCriterion::Gini:
    return std::make_unique<GiniBranchAssignment>(
        assignments, numPartitions, classLeafStats, leafWeights, numClasses);
  case LearningCriterion::SquaredError:
  case LearningCriterion::GainHessian:
  case LearningCriterion::AbsoluteError:
    throw std::invalid_argument(
        "makeClassificationBranchAssignment: criterion is not Entropy or Gini");
  default:
    throw std::invalid_argument(
        "makeClassificationBranchAssignment: unknown criterion");
  }
}

std::unique_ptr<BranchAssignment> makeRegressionBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<float>> &leafFloatData,
    std::vector<double> &leafWeights, double gainHessianLambda) {
  switch (criterion) {
  case LearningCriterion::SquaredError:
    return std::make_unique<SquaredErrorBranchAssignment>(
        assignments, numPartitions, leafFloatData, leafWeights);
  case LearningCriterion::GainHessian:
    return std::make_unique<GainHessianBranchAssignment>(
        assignments, numPartitions, leafFloatData, leafWeights,
        gainHessianLambda);
  case LearningCriterion::AbsoluteError:
    return std::make_unique<AbsoluteErrorBranchAssignment>(
        assignments, numPartitions, leafFloatData, leafWeights);
  case LearningCriterion::Entropy:
  case LearningCriterion::Gini:
    throw std::invalid_argument(
        "makeRegressionBranchAssignment: criterion is not a regression objective");
  default:
    throw std::invalid_argument(
        "makeRegressionBranchAssignment: unknown criterion");
  }
}
