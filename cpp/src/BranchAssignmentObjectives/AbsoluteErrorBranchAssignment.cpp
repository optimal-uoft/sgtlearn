/**
 * @file AbsoluteErrorBranchAssignment.cpp
 * @brief MAE objective with partition medians over raw per-leaf target samples.
 */

#include "AbsoluteErrorBranchAssignment.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

AbsoluteErrorBranchAssignment::AbsoluteErrorBranchAssignment(
    std::vector<size_t> &assignments, size_t numPartitions,
    std::vector<std::vector<float>> &leafYs, std::vector<size_t> &sizes)
    : BranchAssignment(assignments, numPartitions), leafYs_(leafYs),
      sizes_(sizes) {

  if (assignments.size() != leafYs.size() || leafYs.size() != sizes.size())
    throw std::runtime_error(
        "assignments, leafYs, and sizes must have the same length");

  for (size_t i = 0; i < leafYs.size(); ++i) {
    if (leafYs[i].size() != sizes[i])
      throw std::runtime_error("each leafYs[i].size() must equal sizes[i]");
    if (assignments[i] >= numPartitions)
      throw std::runtime_error("assignments[i] must be a valid partition index");
  }

  partitionNumSamples_.assign(numPartitions, 0);
  partitionLoss_.assign(numPartitions, 0.0);

  const size_t numLeaves = assignments.size();
  for (size_t b = 0; b < numLeaves; b++) {
    if (assignments[b] < numPartitions)
      partitionNumSamples_[assignments[b]] += sizes_[b];
  }

  for (size_t p = 0; p < numPartitions; p++) {
    sumNumberOfSamples_ += partitionNumSamples_[p];
    partitionLoss_[p] = computePartitionMae(p);
    weightedSumLoss_ += static_cast<double>(partitionNumSamples_[p]) *
                       partitionLoss_[p];
  }
}

double AbsoluteErrorBranchAssignment::objective() {
  if (!allLeavesAssigned_)
    throw std::runtime_error(
        "Cannot compute objective if any leaves have been unassigned");
  return weightedSumLoss_ / static_cast<double>(sumNumberOfSamples_);
}

void AbsoluteErrorBranchAssignment::addLeaf(size_t leaf, size_t partition) {
  if (allLeavesAssigned_)
    throw std::runtime_error("Cannot assign a leaf if none ever left");

  weightedSumLoss_ -= static_cast<double>(partitionNumSamples_[partition]) *
                     partitionLoss_[partition];

  partitionNumSamples_[partition] += sizes_[leaf];
  sumNumberOfSamples_ += sizes_[leaf];

  partitionLoss_[partition] = computePartitionMae(partition);
  weightedSumLoss_ += static_cast<double>(partitionNumSamples_[partition]) *
                     partitionLoss_[partition];

  assignments[leaf] = partition;
  allLeavesAssigned_ = true;
}

void AbsoluteErrorBranchAssignment::removeLeaf(size_t leaf) {
  if (!allLeavesAssigned_)
    throw std::runtime_error(
        "More than one leaf cannot be removed from the objective");

  const size_t partition = assignments[leaf];
  if (partition >= numPartitions)
    throw std::runtime_error(
        "removeLeaf: leaf is not assigned to a valid partition");

  weightedSumLoss_ -= static_cast<double>(partitionNumSamples_[partition]) *
                     partitionLoss_[partition];
  sumNumberOfSamples_ -= sizes_[leaf];
  partitionNumSamples_[partition] -= sizes_[leaf];

  partitionLoss_[partition] = computePartitionMae(partition);
  weightedSumLoss_ += static_cast<double>(partitionNumSamples_[partition]) *
                     partitionLoss_[partition];

  assignments[leaf] = kUnassignedPartition(numPartitions);
  allLeavesAssigned_ = false;
}

std::vector<float>
AbsoluteErrorBranchAssignment::collectPartitionYs(size_t partition) const {
  size_t total = 0;
  const size_t numLeaves = assignments.size();
  for (size_t b = 0; b < numLeaves; b++) {
    if (assignments[b] < numPartitions && assignments[b] == partition)
      total += sizes_[b];
  }
  std::vector<float> out;
  out.reserve(total);
  for (size_t b = 0; b < numLeaves; b++) {
    if (assignments[b] < numPartitions && assignments[b] == partition)
      out.insert(out.end(), leafYs_[b].begin(), leafYs_[b].end());
  }
  return out;
}

double AbsoluteErrorBranchAssignment::medianAbsoluteError(std::vector<float> ys) {
  if (ys.empty())
    return 0.0;
  std::sort(ys.begin(), ys.end());
  const size_t n = ys.size();
  const double med =
      (n % 2 == 1)
          ? static_cast<double>(ys[n / 2])
          : 0.5 * (static_cast<double>(ys[n / 2 - 1]) + static_cast<double>(ys[n / 2]));
  double sumAbs = 0.0;
  for (float y : ys)
    sumAbs += std::abs(static_cast<double>(y) - med);
  return sumAbs / static_cast<double>(n);
}

double AbsoluteErrorBranchAssignment::computePartitionMae(size_t partition) const {
  if (partitionNumSamples_[partition] == 0)
    return 0.0;
  return medianAbsoluteError(collectPartitionYs(partition));
}
