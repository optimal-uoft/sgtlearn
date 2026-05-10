#include "LeafAggregationBranchAssignment.h"

#include <stdexcept>

namespace leaf_aggregate {

template <typename T>
LeafAggregationBranchAssignment<T>::LeafAggregationBranchAssignment(
    std::vector<size_t> &assignments, size_t numPartitions,
    std::vector<std::vector<T>> &stats, std::vector<size_t> &sizes,
    size_t statsDim, std::unique_ptr<ILeafAggregateProcessor<T>> processor)
    : BranchAssignment(assignments, numPartitions), stats(stats), sizes(sizes),
      statsDim(statsDim), processor_(std::move(processor)) {

  if (assignments.size() != stats.size() || stats.size() != sizes.size())
    throw std::runtime_error("bin statistics must all be the same length");

  for (size_t b = 0; b < assignments.size(); ++b) {
    if (assignments[b] >= numPartitions)
      throw std::runtime_error("assignments[b] must be a valid partition index");
  }

  partitionStats =
      std::vector(numPartitions, std::vector<T>(statsDim, T{}));

  partitionNumSamples = std::vector<size_t>(numPartitions, 0);
  partitionLoss = std::vector<double>(numPartitions, 0.0);

  const size_t numLeaves = assignments.size();
  for (size_t b = 0; b < numLeaves; b++) {
    const size_t partition = assignments[b];
    partitionNumSamples[partition] += sizes[b];
    for (size_t d = 0; d < statsDim; d++)
      partitionStats[partition][d] += stats[b][d];
  }

  for (size_t partition = 0; partition < numPartitions; partition++) {
    sumNumberOfSamples += partitionNumSamples[partition];
    partitionLoss[partition] = computePartitionLoss(partition);
    weightedSumLoss +=
        static_cast<double>(partitionNumSamples[partition]) *
        partitionLoss[partition];
  }
}

template <typename T>
double LeafAggregationBranchAssignment<T>::objective() {
  if (!allLeavesAssigned)
    throw std::runtime_error(
        "Cannot compute objective if any leaves have been unassigned");

  return weightedSumLoss / static_cast<double>(sumNumberOfSamples);
}

template <typename T>
void LeafAggregationBranchAssignment<T>::addLeaf(size_t leaf, size_t partition) {
  if (allLeavesAssigned)
    throw std::runtime_error("Cannot assign a leaf if none ever left");

  weightedSumLoss -= static_cast<double>(partitionNumSamples[partition]) *
                     partitionLoss[partition];

  for (size_t d = 0; d < statsDim; d++)
    partitionStats[partition][d] += stats[leaf][d];
  partitionNumSamples[partition] += sizes[leaf];
  sumNumberOfSamples += sizes[leaf];

  partitionLoss[partition] = computePartitionLoss(partition);
  weightedSumLoss += static_cast<double>(partitionNumSamples[partition]) *
                     partitionLoss[partition];

  assignments[leaf] = partition;
  allLeavesAssigned = true;
}

template <typename T>
void LeafAggregationBranchAssignment<T>::removeLeaf(size_t leaf) {
  if (!allLeavesAssigned)
    throw std::runtime_error(
        "More than one leaf cannot be removed from the objective");

  const size_t partition = assignments[leaf];
  if (partition >= numPartitions)
    throw std::runtime_error(
        "removeLeaf: leaf is not assigned to a valid partition");

  weightedSumLoss -= static_cast<double>(partitionNumSamples[partition]) *
                     partitionLoss[partition];
  sumNumberOfSamples -= sizes[leaf];
  partitionNumSamples[partition] -= sizes[leaf];
  for (size_t d = 0; d < statsDim; d++)
    partitionStats[partition][d] -= stats[leaf][d];

  partitionLoss[partition] = computePartitionLoss(partition);
  weightedSumLoss += static_cast<double>(partitionNumSamples[partition]) *
                     partitionLoss[partition];

  assignments[leaf] = numPartitions;
  allLeavesAssigned = false;
}

template <typename T>
double LeafAggregationBranchAssignment<T>::computePartitionLoss(size_t i) {
  return processor_->compute(partitionStats[i], partitionNumSamples[i]);
}

template class LeafAggregationBranchAssignment<size_t>;
template class LeafAggregationBranchAssignment<float>;

} // namespace leaf_aggregate
