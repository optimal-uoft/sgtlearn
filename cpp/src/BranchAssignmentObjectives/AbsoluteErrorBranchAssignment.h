#pragma once

/**
 * @file AbsoluteErrorBranchAssignment.h
 * @brief MAE branch assignment using per-leaf raw ``y`` samples and partition medians.
 */

#include <cstddef>
#include "BranchAssignment.h"
#include <vector>

/**
 * Multi-output MAE branch-assignment objective: per-partition loss is the SUM
 * over outputs of the MAE about that output's median over the y values in the
 * partition. Holds raw per-leaf, per-output y samples (``leafYs[bin][output]``)
 * with per-sample weights shared across outputs (``leafWs[bin]``); add/remove
 * recomputes the summed MAE for the affected partition(s). Single-output
 * matches the old scalar path.
 */
class AbsoluteErrorBranchAssignment : public BranchAssignment {
public:
  AbsoluteErrorBranchAssignment(
      std::vector<size_t> &assignments, size_t numPartitions,
      std::vector<std::vector<std::vector<float>>> &leafYs,
      std::vector<std::vector<float>> &leafWs,
      std::vector<double> &leafWeights,
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
  bool allLeavesAssigned_ = true;

  std::vector<double> partitionWeight_;
  std::vector<double> partitionLoss_;

  void collectPartitionSamples(size_t partition, size_t output,
                               std::vector<float> &ys,
                               std::vector<float> &ws) const;
  double computePartitionMae(size_t partition) const;

  /** Valid partitions are [0, numPartitions); this marks a leaf not in any partition. */
  static constexpr size_t kUnassignedPartition(size_t numPartitions) {
    return numPartitions;
  }
};
