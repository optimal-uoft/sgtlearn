#pragma once

/**
 * @file AbsoluteErrorBranchAssignmentSort.h
 * @brief Deprecated MAE branch assignment that re-sorts partitions on each move.
 */

#include <cstddef>
#include "BranchAssignment.h"
#include <vector>

/**
 * Reference MAE path: collect partition samples and re-sort on every add/remove.
 *
 * @deprecated Prefer ``AbsoluteErrorBranchAssignment`` (merge/filter). Kept for
 * benchmarks and A/B via ``SGTLEARN_MAE_BACKEND=sort``.
 */
class [[deprecated(
    "Use AbsoluteErrorBranchAssignment (merge/filter); set "
    "SGTLEARN_MAE_BACKEND=sort only for benchmarks")]] AbsoluteErrorBranchAssignmentSort
    : public BranchAssignment {
public:
  AbsoluteErrorBranchAssignmentSort(
      std::vector<size_t> &assignments, size_t numPartitions,
      std::vector<std::vector<std::vector<float>>> &leafYs,
      std::vector<std::vector<float>> &leafWs, std::vector<double> &leafWeights,
      const std::vector<size_t> &leafSampleCounts);

  double objective() override;
  void addLeaf(size_t leaf, size_t partition) override;
  void removeLeaf(size_t leaf) override;

private:
  std::vector<std::vector<std::vector<float>>> &leafYs_;
  std::vector<std::vector<float>> &leafWs_;
  std::vector<double> &leafWeights_;
  size_t nOutputs_ = 0;

  double weightedSumLoss_ = 0;
  double sumNumberOfSamples_ = 0;

  std::vector<double> partitionWeight_;
  std::vector<double> partitionLoss_;

  void collectPartitionSamples(size_t partition, size_t output,
                               std::vector<float> &ys,
                               std::vector<float> &ws) const;
  double computePartitionMae(size_t partition) const;
};
