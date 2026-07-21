/**
 * @file LeafAggregationBranchAssignment.cpp
 * @brief Template method definitions for aggregated branch-assignment objectives.
 */

#include <memory>
#include <utility>
#include <cstddef>
#include "LeafAggregationBranchAssignment.h"

#include <stdexcept>

namespace leaf_aggregate {

template <typename T>
LeafAggregationBranchAssignment<T>::LeafAggregationBranchAssignment(
    std::vector<size_t> &assignments, size_t numPartitions,
    std::vector<std::vector<T>> &stats, std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts, size_t statsDim,
    std::unique_ptr<ILeafAggregateProcessor<T>> processor)
    : BranchAssignment(assignments, numPartitions, leafSampleCounts),
      stats(stats), leafWeights(leafWeights), statsDim(statsDim),
      processor_(std::move(processor)) {

  if (assignments.size() != stats.size() || stats.size() != leafWeights.size())
    throw std::runtime_error("bin statistics must all be the same length");
  if (leafSampleCounts.size() != stats.size())
    throw std::runtime_error(
        "leafSampleCounts must have the same length as bin statistics");

  for (size_t b = 0; b < assignments.size(); ++b) {
    if (assignments[b] >= numPartitions)
      throw std::runtime_error("assignments[b] must be a valid partition index");
  }

  partitionStats =
      std::vector(numPartitions, std::vector<T>(statsDim, T{}));

  partitionWeight = std::vector<double>(numPartitions, 0.0);
  partitionLoss = std::vector<double>(numPartitions, 0.0);

  const size_t numLeaves = assignments.size();
  for (size_t b = 0; b < numLeaves; b++) {
    const size_t partition = assignments[b];
    partitionWeight[partition] += leafWeights[b];
    partitionSampleCount_[partition] += leafSampleCounts_[b];
    for (size_t d = 0; d < statsDim; d++)
      detail::accumulateStat(partitionStats[partition][d], stats[b][d]);
  }

  for (size_t partition = 0; partition < numPartitions; partition++) {
    sumNumberOfSamples += partitionWeight[partition];
    partitionLoss[partition] = computePartitionLoss(partition);
    weightedSumLoss += partitionWeight[partition] * partitionLoss[partition];
  }
}

template <typename T>
double LeafAggregationBranchAssignment<T>::objective() {
  if (!allLeavesAssigned)
    throw std::runtime_error(
        "Cannot compute objective if any leaves have been unassigned");

  return sumNumberOfSamples > 0.0 ? weightedSumLoss / sumNumberOfSamples : 0.0;
}

template <typename T>
void LeafAggregationBranchAssignment<T>::addLeaf(size_t leaf, size_t partition) {
  weightedSumLoss -= partitionWeight[partition] * partitionLoss[partition];

  for (size_t d = 0; d < statsDim; d++)
    detail::accumulateStat(partitionStats[partition][d], stats[leaf][d]);
  partitionWeight[partition] += leafWeights[leaf];
  partitionSampleCount_[partition] += leafSampleCounts_[leaf];
  sumNumberOfSamples += leafWeights[leaf];

  partitionLoss[partition] = computePartitionLoss(partition);
  weightedSumLoss += partitionWeight[partition] * partitionLoss[partition];

  assignments[leaf] = partition;
}

template <typename T>
void LeafAggregationBranchAssignment<T>::removeLeaf(size_t leaf) {
  const size_t partition = assignments[leaf];
  if (partition >= numPartitions)
    throw std::runtime_error(
        "removeLeaf: leaf is not assigned to a valid partition");

  weightedSumLoss -= partitionWeight[partition] * partitionLoss[partition];
  sumNumberOfSamples -= leafWeights[leaf];
  partitionWeight[partition] -= leafWeights[leaf];
  partitionSampleCount_[partition] -= leafSampleCounts_[leaf];
  for (size_t d = 0; d < statsDim; d++)
    detail::subtractStat(partitionStats[partition][d], stats[leaf][d]);

  partitionLoss[partition] = computePartitionLoss(partition);
  weightedSumLoss += partitionWeight[partition] * partitionLoss[partition];

  assignments[leaf] = numPartitions;
}

template <typename T>
double LeafAggregationBranchAssignment<T>::computePartitionLoss(size_t i) {
  return processor_->compute(partitionStats[i], partitionWeight[i]);
}

template class LeafAggregationBranchAssignment<float>;
template class LeafAggregationBranchAssignment<std::vector<double>>;

} // namespace leaf_aggregate
