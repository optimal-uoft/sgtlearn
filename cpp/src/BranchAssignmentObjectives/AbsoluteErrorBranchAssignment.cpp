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
    std::vector<std::vector<float>> &leafYs, std::vector<double> &leafWeights)
    : BranchAssignment(assignments, numPartitions), leafYs_(leafYs),
      leafWeights_(leafWeights) {

  if (assignments.size() != leafYs.size() ||
      leafYs.size() != leafWeights.size())
    throw std::runtime_error(
        "assignments, leafYs, and leafWeights must have the same length");

  for (size_t i = 0; i < leafYs.size(); ++i) {
    if (assignments[i] >= numPartitions)
      throw std::runtime_error("assignments[i] must be a valid partition index");
  }

  partitionWeight_.assign(numPartitions, 0.0);
  partitionLoss_.assign(numPartitions, 0.0);

  const size_t numLeaves = assignments.size();
  for (size_t b = 0; b < numLeaves; b++) {
    if (assignments[b] < numPartitions)
      partitionWeight_[assignments[b]] += leafWeights_[b];
  }

  for (size_t p = 0; p < numPartitions; p++) {
    sumNumberOfSamples_ += partitionWeight_[p];
    partitionLoss_[p] = computePartitionMae(p);
    weightedSumLoss_ += partitionWeight_[p] * partitionLoss_[p];
  }
}

double AbsoluteErrorBranchAssignment::objective() {
  if (!allLeavesAssigned_)
    throw std::runtime_error(
        "Cannot compute objective if any leaves have been unassigned");
  return sumNumberOfSamples_ > 0.0
             ? weightedSumLoss_ / static_cast<double>(sumNumberOfSamples_)
             : 0.0;
}

void AbsoluteErrorBranchAssignment::addLeaf(size_t leaf, size_t partition) {
  if (allLeavesAssigned_)
    throw std::runtime_error("Cannot assign a leaf if none ever left");

  weightedSumLoss_ -= partitionWeight_[partition] * partitionLoss_[partition];

  partitionWeight_[partition] += leafWeights_[leaf];
  sumNumberOfSamples_ += leafWeights_[leaf];

  partitionLoss_[partition] = computePartitionMae(partition);
  weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];

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

  weightedSumLoss_ -= partitionWeight_[partition] * partitionLoss_[partition];
  sumNumberOfSamples_ -= leafWeights_[leaf];
  partitionWeight_[partition] -= leafWeights_[leaf];

  partitionLoss_[partition] = computePartitionMae(partition);
  weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];

  assignments[leaf] = kUnassignedPartition(numPartitions);
  allLeavesAssigned_ = false;
}

std::vector<float>
AbsoluteErrorBranchAssignment::collectPartitionYs(size_t partition) const {
  std::vector<float> ys;
  for (size_t b = 0; b < assignments.size(); ++b) {
    if (assignments[b] != partition)
      continue;
    ys.insert(ys.end(), leafYs_[b].begin(), leafYs_[b].end());
  }
  return ys;
}

double AbsoluteErrorBranchAssignment::medianAbsoluteError(std::vector<float> ys) {
  if (ys.empty())
    return 0.0;
  std::sort(ys.begin(), ys.end());
  const size_t n = ys.size();
  double med;
  if (n % 2 == 1) {
    med = static_cast<double>(ys[n / 2]);
  } else {
    med = 0.5 * (static_cast<double>(ys[n / 2 - 1]) +
                 static_cast<double>(ys[n / 2]));
  }
  double s = 0.0;
  for (float v : ys)
    s += std::fabs(static_cast<double>(v) - med);
  return s / static_cast<double>(n);
}

double AbsoluteErrorBranchAssignment::computePartitionMae(size_t partition) const {
  return medianAbsoluteError(collectPartitionYs(partition));
}
