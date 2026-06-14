#pragma once

/**
 * @file BranchAssignment.h
 * @brief Abstract objective for mapping inner discretizer bins to a fixed number of child partitions (coordinate-descent moves ``assignments``).
 */

#include <cstddef>
#include <vector>

/** Mutable bin-to-partition map with add/remove hooks for incremental objective evaluation. */
class BranchAssignment {
public:
  BranchAssignment(std::vector<size_t> &assignments, size_t numPartitions,
                   const std::vector<size_t> &leafSampleCounts)
      : assignments(assignments), numPartitions(numPartitions),
        leafSampleCounts_(leafSampleCounts),
        partitionSampleCount_(numPartitions, 0) {}

  virtual ~BranchAssignment() = default;

  std::vector<size_t> &assignments;
  size_t numPartitions;

  virtual double objective() = 0;

  virtual void addLeaf(size_t leaf, size_t partition) = 0;

  virtual void removeLeaf(size_t leaf) = 0;

  /** Unweighted training sample count per child partition (sum of inner bin ``N``). */
  const std::vector<size_t> &partitionSampleCounts() const {
    return partitionSampleCount_;
  }

  /** True when every partition holds at least ``minLeafSize`` samples. */
  bool partitionCountsMeetMinLeaf(size_t minLeafSize) const {
    for (size_t p = 0; p < numPartitions; ++p) {
      if (partitionSampleCount_[p] < minLeafSize)
        return false;
    }
    return true;
  }

protected:
  const std::vector<size_t> &leafSampleCounts_;
  std::vector<size_t> partitionSampleCount_;
};
