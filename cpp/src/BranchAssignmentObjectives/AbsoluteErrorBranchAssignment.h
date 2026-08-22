#pragma once

/**
 * @file AbsoluteErrorBranchAssignment.h
 * @brief Default MAE branch assignment: sorted bins + merge/filter partitions.
 *
 * Production AbsoluteError backend. Deprecated alternatives:
 * ``AbsoluteErrorBranchAssignmentBst``, ``AbsoluteErrorBranchAssignmentSort``.
 * Hot-swap via ``SGTLEARN_MAE_BACKEND`` (default ``merge``).
 */

#include <cstddef>
#include "BranchAssignment.h"
#include <vector>

/**
 * Multi-output MAE using pre-sorted per-bin arrays and sorted partitions.
 * Join: mergesort-style merge (``O(n + k)``). Leave: filter by source-bin id
 * (``O(n)``). MAE uses ``Criterion::absoluteErrorPresorted``.
 */
class AbsoluteErrorBranchAssignment : public BranchAssignment {
public:
  AbsoluteErrorBranchAssignment(
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

  std::vector<std::vector<std::vector<float>>> binYsSorted_;
  std::vector<std::vector<std::vector<float>>> binWsSorted_;

  std::vector<std::vector<std::vector<float>>> partYs_;
  std::vector<std::vector<std::vector<float>>> partWs_;
  std::vector<std::vector<std::vector<size_t>>> partSrcBin_;

  void buildSortedBin(size_t leaf);
  void mergeLeafIntoPartition(size_t leaf, size_t partition);
  void filterLeafFromPartition(size_t leaf, size_t partition);
  double computePartitionMae(size_t partition) const;
};

/** @deprecated Prefer ``AbsoluteErrorBranchAssignment`` (merge is default). */
using AbsoluteErrorBranchAssignmentMerge [[deprecated(
    "Use AbsoluteErrorBranchAssignment; merge is the default backend")]] =
    AbsoluteErrorBranchAssignment;
