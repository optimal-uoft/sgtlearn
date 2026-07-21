#pragma once

/**
 * @file LeafAggregationBranchAssignment.h
 * @brief Template ``BranchAssignment`` that aggregates per-leaf stats into partitions via an ``ILeafAggregateProcessor``.
 */

#include <cstddef>
#include "BranchAssignment.h"
#include "LeafAggregateProcessor.h"
#include <memory>
#include <vector>

namespace leaf_aggregate {

namespace detail {

template <typename T>
inline void accumulateStat(T &dst, const T &src) {
  dst += src;
}

template <typename T>
inline void subtractStat(T &dst, const T &src) {
  dst -= src;
}

inline void accumulateStat(std::vector<double> &dst,
                           const std::vector<double> &src) {
  if (dst.size() < src.size())
    dst.resize(src.size(), 0.0);
  for (size_t i = 0; i < src.size(); ++i)
    dst[i] += src[i];
}

inline void subtractStat(std::vector<double> &dst,
                         const std::vector<double> &src) {
  if (dst.size() < src.size())
    dst.resize(src.size(), 0.0);
  for (size_t i = 0; i < src.size(); ++i)
    dst[i] -= src[i];
}

} // namespace detail

/**
 * Coordinate-descent objective for assigning histogram leaves to partitions.
 *
 * @tparam T bin statistic scalar type (e.g. ``double`` for regression power
 *         sums, ``std::vector<double>`` for nested class histograms).
 */
template <typename T>
class LeafAggregationBranchAssignment : public BranchAssignment {

public:
  ~LeafAggregationBranchAssignment() override = default;

  LeafAggregationBranchAssignment(std::vector<size_t> &assignments,
                                  size_t numPartitions,
                                  std::vector<std::vector<T>> &stats,
                                  std::vector<double> &leafWeights,
                                  const std::vector<size_t> &leafSampleCounts,
                                  size_t statsDim,
                                  std::unique_ptr<ILeafAggregateProcessor<T>> processor);

  double objective() override;

  void addLeaf(size_t leaf, size_t partition) override;

  void removeLeaf(size_t leaf) override;

  const std::vector<std::vector<T>> &aggregatedPartitionStats() const {
    return partitionStats;
  }

  const std::vector<double> &aggregatedPartitionWeights() const {
    return partitionWeight;
  }

protected:
  double weightedSumLoss = 0;
  double sumNumberOfSamples = 0;
  bool allLeavesAssigned = true;

  std::vector<std::vector<T>> &stats;
  const std::vector<double> &leafWeights;
  size_t statsDim;

  std::vector<std::vector<T>> partitionStats;
  std::vector<double> partitionWeight;
  std::vector<double> partitionLoss;

  std::unique_ptr<ILeafAggregateProcessor<T>> processor_;

  double computePartitionLoss(size_t i);
};

extern template class LeafAggregationBranchAssignment<float>;
extern template class LeafAggregationBranchAssignment<std::vector<double>>;

} // namespace leaf_aggregate
