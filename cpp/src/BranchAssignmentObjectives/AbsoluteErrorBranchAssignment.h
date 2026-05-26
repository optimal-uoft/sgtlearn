#pragma once

/**
 * @file AbsoluteErrorBranchAssignment.h
 * @brief MAE branch assignment using per-leaf raw ``y`` samples and partition medians.
 */

#include "BranchAssignment.h"
#include <vector>

/**
 * Branch-assignment objective: per-partition loss is MAE about the median of all
 * y values in that partition (same notion as AbsoluteErrorSplitter scoring).
 * Holds raw per-leaf y samples; add/remove recomputes median and MAE for the
 * affected partition(s).
 */
class AbsoluteErrorBranchAssignment : public BranchAssignment {
public:
  AbsoluteErrorBranchAssignment(std::vector<size_t> &assignments,
                                size_t numPartitions,
                                std::vector<std::vector<float>> &leafYs,
                                std::vector<std::vector<float>> &leafWs,
                                std::vector<double> &leafWeights);

  double objective() override;

  void addLeaf(size_t leaf, size_t partition) override;

  void removeLeaf(size_t leaf) override;

private:
  std::vector<std::vector<float>> &leafYs_;
  std::vector<std::vector<float>> &leafWs_;
  std::vector<double> &leafWeights_;

  double weightedSumLoss_ = 0;
  double sumNumberOfSamples_ = 0;
  bool allLeavesAssigned_ = true;

  std::vector<double> partitionWeight_;
  std::vector<double> partitionLoss_;

  void collectPartitionSamples(size_t partition, std::vector<float> &ys,
                               std::vector<float> &ws) const;
  double computePartitionMae(size_t partition) const;

  /** Valid partitions are [0, numPartitions); this marks a leaf not in any partition. */
  static constexpr size_t kUnassignedPartition(size_t numPartitions) {
    return numPartitions;
  }
};
