/**
 * @file AbsoluteErrorBranchAssignmentSort.cpp
 * @brief Deprecated full re-sort MAE branch assignment.
 */

#include "AbsoluteErrorBranchAssignmentSort.h"

#include "AbsoluteErrorBranchAssignmentCommon.h"
#include "Criterion.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

AbsoluteErrorBranchAssignmentSort::AbsoluteErrorBranchAssignmentSort(
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

  const size_t numLeaves = assignments.size();
  for (size_t b = 0; b < numLeaves; ++b) {
    if (assignments[b] < numPartitions) {
      partitionWeight_[assignments[b]] += leafWeights_[b];
      partitionSampleCount_[assignments[b]] += leafSampleCounts_[b];
    }
  }

  for (size_t p = 0; p < numPartitions; ++p) {
    sumNumberOfSamples_ += partitionWeight_[p];
    partitionLoss_[p] = computePartitionMae(p);
    weightedSumLoss_ += partitionWeight_[p] * partitionLoss_[p];
  }
}

double AbsoluteErrorBranchAssignmentSort::objective() {
  return sumNumberOfSamples_ > 0.0
             ? weightedSumLoss_ / static_cast<double>(sumNumberOfSamples_)
             : 0.0;
}

void AbsoluteErrorBranchAssignmentSort::addLeaf(size_t leaf, size_t partition) {
  weightedSumLoss_ -= partitionWeight_[partition] * partitionLoss_[partition];

  partitionWeight_[partition] += leafWeights_[leaf];
  partitionSampleCount_[partition] += leafSampleCounts_[leaf];
  sumNumberOfSamples_ += leafWeights_[leaf];
  assignments[leaf] = partition;

  partitionLoss_[partition] = computePartitionMae(partition);
  weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];
}

void AbsoluteErrorBranchAssignmentSort::removeLeaf(size_t leaf) {
  const size_t partition = assignments[leaf];
  if (partition >= numPartitions)
    throw std::runtime_error(
        "removeLeaf: leaf is not assigned to a valid partition");

  weightedSumLoss_ -= partitionWeight_[partition] * partitionLoss_[partition];
  sumNumberOfSamples_ -= leafWeights_[leaf];
  partitionWeight_[partition] -= leafWeights_[leaf];
  partitionSampleCount_[partition] -= leafSampleCounts_[leaf];
  assignments[leaf] = absolute_error_branch::unassignedPartition(numPartitions);

  partitionLoss_[partition] = computePartitionMae(partition);
  weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];
}

void AbsoluteErrorBranchAssignmentSort::collectPartitionSamples(
    size_t partition, size_t output, std::vector<float> &ys,
    std::vector<float> &ws) const {
  ys.clear();
  ws.clear();
  for (size_t b = 0; b < assignments.size(); ++b) {
    if (assignments[b] != partition)
      continue;
    if (output < leafYs_[b].size())
      ys.insert(ys.end(), leafYs_[b][output].begin(), leafYs_[b][output].end());
    ws.insert(ws.end(), leafWs_[b].begin(), leafWs_[b].end());
  }
}

double
AbsoluteErrorBranchAssignmentSort::computePartitionMae(size_t partition) const {
  double total = 0.0;
  std::vector<float> ys;
  std::vector<float> ws;
  for (size_t o = 0; o < nOutputs_; ++o) {
    collectPartitionSamples(partition, o, ys, ws);
    if (ys.size() <= 1) {
      total += Criterion::absoluteError(ys, ws).mae;
      continue;
    }
    std::vector<size_t> order(ys.size());
    for (size_t i = 0; i < order.size(); ++i)
      order[i] = i;
    std::sort(order.begin(), order.end(),
              [&ys](size_t a, size_t b) { return ys[a] < ys[b]; });
    std::vector<float> ysSorted;
    std::vector<float> wsSorted;
    ysSorted.reserve(ys.size());
    wsSorted.reserve(ws.size());
    for (size_t idx : order) {
      ysSorted.push_back(ys[idx]);
      wsSorted.push_back(ws[idx]);
    }
    total += Criterion::absoluteErrorPresorted(ysSorted, wsSorted).mae;
  }
  return total;
}

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
