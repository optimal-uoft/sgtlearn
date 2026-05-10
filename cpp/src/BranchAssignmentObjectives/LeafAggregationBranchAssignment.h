#pragma once

#include "BranchAssignment.h"
#include <concepts>
#include <vector>

template <typename Fn, typename T>
concept CriterionFunction = requires(Fn f, const std::vector<T> &v, size_t n) {
  { f(v, n) } -> std::convertible_to<double>;
};

template <typename T, auto CriterionFn>
  requires CriterionFunction<decltype(CriterionFn), T>
class LeafAggregationBranchAssignment : public BranchAssignment {

public:
  ~LeafAggregationBranchAssignment() override = default;

  LeafAggregationBranchAssignment(std::vector<size_t> &assignments,
                                  size_t numPartitions,
                                  std::vector<std::vector<T>> &stats,
                                  std::vector<size_t> &sizes, size_t statsDim);

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

  double computePartitionLoss(size_t i);
};
