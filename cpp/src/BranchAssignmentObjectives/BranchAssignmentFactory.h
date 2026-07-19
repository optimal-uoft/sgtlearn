#pragma once

/**
 * @file BranchAssignmentFactory.h
 * @brief Factory that constructs the correct ``BranchAssignment`` for a ``LearningCriterion``.
 */

#include <cstddef>
#include "BranchAssignment.h"
#include "Domain/LearningCriterion.h"

#include <memory>
#include <vector>

/**
 * Build a branch-assignment objective for @p criterion.
 *
 * Classification (Entropy, Gini): pass @p classesPerOutput and per-leaf
 * concatenated class-count stats in @p leafStats.
 *
 * SquaredError: pass per-leaf ``[Σw·y0, Σw·y0², ...]`` stats (length
 * ``2 * nOutputs``) in @p leafStats and set @p nOutputs.
 *
 * AbsoluteError: pass raw per-leaf, per-output y samples and per-sample weights
 * via @p maeLeafYs (``[bin][output][sample]``) and @p maeLeafWs
 * (``[bin][sample]``); both required.
 */
std::unique_ptr<BranchAssignment> makeBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<double>> &leafStats,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts,
    const std::vector<size_t> &classesPerOutput = {}, size_t nOutputs = 1,
    std::vector<std::vector<std::vector<float>>> *maeLeafYs = nullptr,
    std::vector<std::vector<float>> *maeLeafWs = nullptr);
