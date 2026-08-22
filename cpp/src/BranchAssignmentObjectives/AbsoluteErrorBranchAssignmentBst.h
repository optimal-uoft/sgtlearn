#pragma once

/**
 * @file AbsoluteErrorBranchAssignmentBst.h
 * @brief Deprecated MAE branch assignment using per-partition WeightedMAETree.
 */

#include <cstddef>
#include "BranchAssignment.h"
#include "algorithms/WeightedMAETree.h"
#include <vector>

/**
 * Multi-output MAE with per-partition ``WeightedMAETree`` multisets.
 *
 * @deprecated Prefer ``AbsoluteErrorBranchAssignment`` (merge/filter). Kept for
 * benchmarks and A/B via ``SGTLEARN_MAE_BACKEND=bst``.
 */
class [[deprecated(
    "Use AbsoluteErrorBranchAssignment (merge/filter); set "
    "SGTLEARN_MAE_BACKEND=bst only for benchmarks")]] AbsoluteErrorBranchAssignmentBst
    : public BranchAssignment {
public:
  AbsoluteErrorBranchAssignmentBst(
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
  std::vector<std::vector<WeightedMAETree>> trees_;

  double computePartitionMae(size_t partition) const;
  void insertLeafIntoPartition(size_t leaf, size_t partition);
  void eraseLeafFromPartition(size_t leaf, size_t partition);
};
