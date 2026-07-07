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
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts, size_t numClasses);

/**
 * SquaredError or AbsoluteError only.
 *
 * For SquaredError, @p leafRegressionStats is per-leaf aggregated statistics.
 * For AbsoluteError, @p maeLeafYs and @p maeLeafWs are raw y samples and
 * matching per-sample weights per leaf (required for AbsoluteError).
 */
std::unique_ptr<BranchAssignment> makeRegressionBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<double>> &leafRegressionStats,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts,
    std::vector<std::vector<float>> *maeLeafYs = nullptr,
    std::vector<std::vector<float>> *maeLeafWs = nullptr);
