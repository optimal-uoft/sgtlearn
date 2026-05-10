#pragma once

#include "BranchAssignment.h"
#include "LeafAggregateProcessor.h"
#include <memory>
#include <vector>

namespace leaf_aggregate {

/**
 * Coordinate-descent objective for assigning histogram leaves to partitions.
 *
 * @tparam T bin statistic scalar type (e.g. size_t for class counts, float for gradients).
 */
template <typename T>
class LeafAggregationBranchAssignment : public BranchAssignment {

public:
  ~LeafAggregationBranchAssignment() override = default;

  LeafAggregationBranchAssignment(std::vector<size_t> &assignments,
                                  size_t numPartitions,
                                  std::vector<std::vector<T>> &stats,
                                  std::vector<size_t> &sizes, size_t statsDim,
                                  std::unique_ptr<ILeafAggregateProcessor<T>> processor);

  double objective() override;

  void addLeaf(size_t leaf, size_t partition) override;

  void removeLeaf(size_t leaf) override;

protected:
  double weightedSumLoss = 0;
  size_t sumNumberOfSamples = 0;
  bool allLeavesAssigned = true;

  std::vector<std::vector<T>> &stats;
  const std::vector<size_t> &sizes;
  size_t statsDim;

  std::vector<std::vector<T>> partitionStats;
  std::vector<size_t> partitionNumSamples;
  std::vector<double> partitionLoss;

  std::unique_ptr<ILeafAggregateProcessor<T>> processor_;

  double computePartitionLoss(size_t i);
};

extern template class LeafAggregationBranchAssignment<size_t>;
extern template class LeafAggregationBranchAssignment<float>;

} // namespace leaf_aggregate
