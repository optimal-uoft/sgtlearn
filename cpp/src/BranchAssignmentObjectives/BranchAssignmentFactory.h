#pragma once

/**
 * @file BranchAssignmentFactory.h
 * @brief Factory that constructs the correct ``BranchAssignment`` for a ``LearningCriterion``.
 */

#include "BranchAssignment.h"
#include "Domain/LearningCriterion.h"

#include <memory>
#include <vector>

/**
 * Build a branch-assignment objective for @p criterion.
 *
 * Classification (Entropy, Gini): pass @p numClasses and per-leaf class-count
 * stats in @p leafStats.
 *
 * SquaredError: pass per-leaf ``{sum y, sum y^2}`` stats in @p leafStats.
 *
 * AbsoluteError: pass raw per-leaf y samples and weights via @p maeLeafYs and
 * @p maeLeafWs (required).
 */
std::unique_ptr<BranchAssignment> makeBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<double>> &leafStats,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts, size_t numClasses = 0,
    std::vector<std::vector<float>> *maeLeafYs = nullptr,
    std::vector<std::vector<float>> *maeLeafWs = nullptr);
