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
 * Regression MAE branch assignment.
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
    std::vector<std::vector<std::vector<float>>> *maeLeafYs,
    std::vector<std::vector<float>> *maeLeafWs);

/**
 * Classification / regression MSE branch assignment.
 *
 * Pass nested per-leaf stats ``[bin][output][*]`` in @p leafStats. For
 * classification, ``*`` is class counts and @p classesPerOutput is required.
 * For regression MSE, ``*`` is ``[Σw·y, Σw·y²]`` and @p nOutputs must be set.
 */
std::unique_ptr<BranchAssignment> makeBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions,
    std::vector<std::vector<std::vector<double>>> &leafStats,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts,
    const std::vector<size_t> &classesPerOutput = {}, size_t nOutputs = 1);
