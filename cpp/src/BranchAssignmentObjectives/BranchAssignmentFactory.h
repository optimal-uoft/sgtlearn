#pragma once

/**
 * @file BranchAssignmentFactory.h
 * @brief Factory functions that construct the correct ``BranchAssignment`` for a ``LearningCriterion``.
 */

#include "BranchAssignment.h"
#include "Domain/LearningCriterion.h"

#include <memory>
#include <vector>

/** Entropy or Gini only. Throws if criterion is not classification. */
std::unique_ptr<BranchAssignment> makeClassificationBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<double>> &classLeafStats,
    std::vector<double> &leafWeights, size_t numClasses);

/**
 * SquaredError, GainHessian, or AbsoluteError only.
 *
 * For SquaredError / GainHessian, @p leafFloatData is per-leaf aggregated
 * statistics. For AbsoluteError, it is raw y samples per leaf and
 * @p leafSampleWeights holds matching per-sample weights.
 *
 * @param gainHessianLambda used only for GainHessian.
 */
std::unique_ptr<BranchAssignment> makeRegressionBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<double>> &leafRegressionStats,
    std::vector<double> &leafWeights, double gainHessianLambda = 1.0,
    std::vector<std::vector<float>> *maeLeafYs = nullptr,
    std::vector<std::vector<float>> *maeLeafWs = nullptr);
