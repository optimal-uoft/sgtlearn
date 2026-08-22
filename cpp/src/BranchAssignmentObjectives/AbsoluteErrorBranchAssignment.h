#pragma once

/**
 * @file AbsoluteErrorBranchAssignment.h
 * @brief MAE branch assignment with per-partition ``WeightedMAETree`` multisets.
 */

#include <cstddef>
#include "BranchAssignment.h"
#include "algorithms/WeightedMAETree.h"
#include <vector>

/**
 * Multi-output MAE branch-assignment objective: per-partition loss is the SUM
 * over outputs of the MAE about that output's median. Each partition/output
 * owns a ``WeightedMAETree``; ``addLeaf`` / ``removeLeaf`` batch-insert or
 * batch-erase that bin's ``(y, w)`` samples in ``O(K log N)``.
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

  std::vector<double> partitionWeight_;
  std::vector<double> partitionLoss_;
  /** ``trees_[partition][output]``. */
  std::vector<std::vector<WeightedMAETree>> trees_;

  double computePartitionMae(size_t partition) const;

  void insertLeafIntoPartition(size_t leaf, size_t partition);
  void eraseLeafFromPartition(size_t leaf, size_t partition);

  /** Valid partitions are [0, numPartitions); this marks a leaf not in any partition. */
  static constexpr size_t kUnassignedPartition(size_t numPartitions) {
    return numPartitions;
  }
};

/**
 * Reference MAE branch assignment that re-sorts each affected partition on every
 * add/remove (original ``O(n log n)`` path). Kept for correctness / perf tests.
 */
class AbsoluteErrorBranchAssignmentSort : public BranchAssignment {
public:
  AbsoluteErrorBranchAssignmentSort(
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

  std::vector<double> partitionWeight_;
  std::vector<double> partitionLoss_;

  void collectPartitionSamples(size_t partition, size_t output,
                               std::vector<float> &ys,
                               std::vector<float> &ws) const;
  double computePartitionMae(size_t partition) const;

  static constexpr size_t kUnassignedPartition(size_t numPartitions) {
    return numPartitions;
  }
};
