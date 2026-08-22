#pragma once

/**
 * @file AbsoluteErrorBranchAssignmentCommon.h
 * @brief Shared validation helpers for AbsoluteError branch-assignment backends.
 */

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace absolute_error_branch {

inline constexpr size_t unassignedPartition(size_t numPartitions) {
  return numPartitions;
}

inline void validateInputs(
    const std::vector<size_t> &assignments, size_t numPartitions,
    const std::vector<std::vector<std::vector<float>>> &leafYs,
    const std::vector<std::vector<float>> &leafWs,
    const std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts, size_t &nOutputs) {
  if (assignments.size() != leafYs.size() || leafYs.size() != leafWs.size() ||
      leafYs.size() != leafWeights.size())
    throw std::runtime_error(
        "assignments, leafYs, leafWs, and leafWeights must have the same length");
  if (leafSampleCounts.size() != leafYs.size())
    throw std::runtime_error(
        "leafSampleCounts must have the same length as bin statistics");

  nOutputs = 0;
  for (const auto &binOutputs : leafYs) {
    if (!binOutputs.empty()) {
      nOutputs = binOutputs.size();
      break;
    }
  }

  for (size_t i = 0; i < leafYs.size(); ++i) {
    if (assignments[i] >= numPartitions)
      throw std::runtime_error("assignments[i] must be a valid partition index");
    for (const auto &outputYs : leafYs[i]) {
      if (outputYs.size() != leafWs[i].size())
        throw std::runtime_error(
            "leafYs[i][o] and leafWs[i] must have the same length");
    }
  }
}

} // namespace absolute_error_branch
