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
                                std::vector<size_t> &sizes);

  double objective() override;

  void addLeaf(size_t leaf, size_t partition) override;

  void removeLeaf(size_t leaf) override;

private:
  std::vector<std::vector<float>> &leafYs_;
  std::vector<size_t> &sizes_;

  double weightedSumLoss_ = 0;
  size_t sumNumberOfSamples_ = 0;
  bool allLeavesAssigned_ = true;

  std::vector<size_t> partitionNumSamples_;
  std::vector<double> partitionLoss_;

  std::vector<float> collectPartitionYs(size_t partition) const;
  static double medianAbsoluteError(std::vector<float> ys);
  double computePartitionMae(size_t partition) const;

  /** Valid partitions are [0, numPartitions); this marks a leaf not in any partition. */
  static constexpr size_t kUnassignedPartition(size_t numPartitions) {
    return numPartitions;
  }
};
