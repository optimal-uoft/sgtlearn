/**
 * @file AbsoluteErrorBranchAssignmentBst.cpp
 * @brief Deprecated AVL/order-statistic MAE branch assignment.
 */

#include "AbsoluteErrorBranchAssignmentBst.h"

#include "AbsoluteErrorBranchAssignmentCommon.h"

#include <cstddef>
#include <stdexcept>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

AbsoluteErrorBranchAssignmentBst::AbsoluteErrorBranchAssignmentBst(
    std::vector<size_t> &assignments, size_t numPartitions,
    std::vector<std::vector<std::vector<float>>> &leafYs,
    std::vector<std::vector<float>> &leafWs, std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts)
    : BranchAssignment(assignments, numPartitions, leafSampleCounts),
      leafYs_(leafYs), leafWs_(leafWs), leafWeights_(leafWeights) {

  absolute_error_branch::validateInputs(assignments, numPartitions, leafYs,
                                        leafWs, leafWeights, leafSampleCounts,
                                        nOutputs_);

  partitionWeight_.assign(numPartitions, 0.0);
  partitionLoss_.assign(numPartitions, 0.0);
  trees_.resize(numPartitions);
  for (size_t p = 0; p < numPartitions; ++p)
    trees_[p].resize(nOutputs_);

  const size_t numLeaves = assignments.size();
  for (size_t b = 0; b < numLeaves; ++b) {
    if (assignments[b] >= numPartitions)
      continue;
    partitionWeight_[assignments[b]] += leafWeights_[b];
    partitionSampleCount_[assignments[b]] += leafSampleCounts_[b];
    insertLeafIntoPartition(b, assignments[b]);
  }

  for (size_t p = 0; p < numPartitions; ++p) {
    sumNumberOfSamples_ += partitionWeight_[p];
    partitionLoss_[p] = computePartitionMae(p);
    weightedSumLoss_ += partitionWeight_[p] * partitionLoss_[p];
  }
}

double AbsoluteErrorBranchAssignmentBst::objective() {
  return sumNumberOfSamples_ > 0.0
             ? weightedSumLoss_ / static_cast<double>(sumNumberOfSamples_)
             : 0.0;
}

void AbsoluteErrorBranchAssignmentBst::insertLeafIntoPartition(
    size_t leaf, size_t partition) {
  if (nOutputs_ == 0)
    return;
  const auto &ws = leafWs_[leaf];
  for (size_t o = 0; o < nOutputs_; ++o) {
    if (o >= leafYs_[leaf].size())
      continue;
    trees_[partition][o].insert_batch(leafYs_[leaf][o], ws);
  }
}

void AbsoluteErrorBranchAssignmentBst::eraseLeafFromPartition(
    size_t leaf, size_t partition) {
  if (nOutputs_ == 0)
    return;
  const auto &ws = leafWs_[leaf];
  for (size_t o = 0; o < nOutputs_; ++o) {
    if (o >= leafYs_[leaf].size())
      continue;
    trees_[partition][o].remove_batch(leafYs_[leaf][o], ws);
  }
}

void AbsoluteErrorBranchAssignmentBst::addLeaf(size_t leaf, size_t partition) {
  weightedSumLoss_ -= partitionWeight_[partition] * partitionLoss_[partition];

  partitionWeight_[partition] += leafWeights_[leaf];
  partitionSampleCount_[partition] += leafSampleCounts_[leaf];
  sumNumberOfSamples_ += leafWeights_[leaf];

  insertLeafIntoPartition(leaf, partition);

  partitionLoss_[partition] = computePartitionMae(partition);
  weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];

  assignments[leaf] = partition;
}

void AbsoluteErrorBranchAssignmentBst::removeLeaf(size_t leaf) {
  const size_t partition = assignments[leaf];
  if (partition >= numPartitions)
    throw std::runtime_error(
        "removeLeaf: leaf is not assigned to a valid partition");

  weightedSumLoss_ -= partitionWeight_[partition] * partitionLoss_[partition];
  sumNumberOfSamples_ -= leafWeights_[leaf];
  partitionWeight_[partition] -= leafWeights_[leaf];
  partitionSampleCount_[partition] -= leafSampleCounts_[leaf];

  eraseLeafFromPartition(leaf, partition);

  partitionLoss_[partition] = computePartitionMae(partition);
  weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];

  assignments[leaf] = absolute_error_branch::unassignedPartition(numPartitions);
}

double
AbsoluteErrorBranchAssignmentBst::computePartitionMae(size_t partition) const {
  double total = 0.0;
  for (size_t o = 0; o < nOutputs_; ++o)
    total += trees_[partition][o].mae();
  return total;
}

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
