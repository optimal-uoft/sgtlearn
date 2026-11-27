#pragma once

/**
 * @file LeafAggregationBranchAssignment.h
 * @brief ``BranchAssignment`` that aggregates per-leaf stats into partitions for
 *        entropy, Gini, and squared-error objectives.
 */

#include <cstddef>
#include <vector>

#include "BranchAssignment.h"
#include "Domain/LearningCriterion.h"

/**
 * Coordinate-descent objective for assigning histogram / moment leaves to
 * partitions.
 *
 * ``stats[leaf][output]`` holds per-output class histograms (entropy / Gini)
 * or ``[Σw·y, Σw·y²]`` moments (squared error).
 */
class LeafAggregationBranchAssignment : public BranchAssignment {
public:
  ~LeafAggregationBranchAssignment() override = default;

  LeafAggregationBranchAssignment(
      LearningCriterion criterion, std::vector<size_t> &assignments,
      size_t numPartitions,
      std::vector<std::vector<std::vector<double>>> &stats,
      std::vector<double> &leafWeights,
      const std::vector<size_t> &leafSampleCounts, size_t statsDim);

  double objective() override;
  void addLeaf(size_t leaf, size_t partition) override;
  void removeLeaf(size_t leaf) override;

  const std::vector<std::vector<std::vector<double>>> &
  aggregatedPartitionStats() const {
    return partitionStats_;
  }

  const std::vector<double> &aggregatedPartitionWeights() const {
    return partitionWeight_;
  }

private:
  LearningCriterion criterion_;
  double weightedSumLoss_ = 0.0;
  double sumNumberOfSamples_ = 0.0;

  std::vector<std::vector<std::vector<double>>> &stats_;
  const std::vector<double> &leafWeights_;
  size_t statsDim_;

  std::vector<std::vector<std::vector<double>>> partitionStats_;
  std::vector<double> partitionWeight_;
  std::vector<double> partitionLoss_;

  double computePartitionLoss(size_t i) const;
};
