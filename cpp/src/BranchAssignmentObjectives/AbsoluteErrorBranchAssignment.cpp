/**
 * @file AbsoluteErrorBranchAssignment.cpp
 * @brief Default MAE branch assignment (sorted merge / filter).
 */

#include "AbsoluteErrorBranchAssignment.h"

#include "AbsoluteErrorBranchAssignmentCommon.h"
#include "Criterion.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

AbsoluteErrorBranchAssignment::AbsoluteErrorBranchAssignment(
    std::vector<size_t> &assignments, size_t numPartitions,
    std::vector<std::vector<std::vector<float>>> &leafYs,
    std::vector<std::vector<float>> &leafWs, std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts)
    : BranchAssignment(assignments, numPartitions, leafSampleCounts),
      leafYs_(leafYs), leafWs_(leafWs), leafWeights_(leafWeights) {

  absolute_error_branch::validateInputs(assignments, numPartitions, leafYs,
                                        leafWs, leafWeights, leafSampleCounts,
                                        nOutputs_);

  const size_t numLeaves = assignments.size();
  binYsSorted_.assign(numLeaves, {});
  binWsSorted_.assign(numLeaves, {});
  for (size_t b = 0; b < numLeaves; ++b)
    buildSortedBin(b);

  partitionWeight_.assign(numPartitions, 0.0);
  partitionLoss_.assign(numPartitions, 0.0);
  partYs_.assign(numPartitions, std::vector<std::vector<float>>(nOutputs_));
  partWs_.assign(numPartitions, std::vector<std::vector<float>>(nOutputs_));
  partSrcBin_.assign(numPartitions,
                     std::vector<std::vector<size_t>>(nOutputs_));

  for (size_t b = 0; b < numLeaves; ++b) {
    if (assignments[b] >= numPartitions)
      continue;
    partitionWeight_[assignments[b]] += leafWeights_[b];
    partitionSampleCount_[assignments[b]] += leafSampleCounts_[b];
    mergeLeafIntoPartition(b, assignments[b]);
  }

  for (size_t p = 0; p < numPartitions; ++p) {
    sumNumberOfSamples_ += partitionWeight_[p];
    partitionLoss_[p] = computePartitionMae(p);
    weightedSumLoss_ += partitionWeight_[p] * partitionLoss_[p];
  }
}

void AbsoluteErrorBranchAssignment::buildSortedBin(size_t leaf) {
  binYsSorted_[leaf].assign(nOutputs_, {});
  binWsSorted_[leaf].assign(nOutputs_, {});
  const auto &ws = leafWs_[leaf];
  for (size_t o = 0; o < nOutputs_; ++o) {
    if (o >= leafYs_[leaf].size())
      continue;
    const auto &ys = leafYs_[leaf][o];
    const size_t n = ys.size();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i)
      order[i] = i;
    std::sort(order.begin(), order.end(),
              [&ys](size_t a, size_t b) { return ys[a] < ys[b]; });
    auto &ysOut = binYsSorted_[leaf][o];
    auto &wsOut = binWsSorted_[leaf][o];
    ysOut.resize(n);
    wsOut.resize(n);
    for (size_t i = 0; i < n; ++i) {
      ysOut[i] = ys[order[i]];
      wsOut[i] = ws[order[i]];
    }
  }
}

void AbsoluteErrorBranchAssignment::mergeLeafIntoPartition(size_t leaf,
                                                           size_t partition) {
  for (size_t o = 0; o < nOutputs_; ++o) {
    const auto &binY = binYsSorted_[leaf][o];
    const auto &binW = binWsSorted_[leaf][o];
    auto &partY = partYs_[partition][o];
    auto &partW = partWs_[partition][o];
    auto &partSrc = partSrcBin_[partition][o];

    if (binY.empty())
      continue;
    if (partY.empty()) {
      partY = binY;
      partW = binW;
      partSrc.assign(binY.size(), leaf);
      continue;
    }

    std::vector<float> outY;
    std::vector<float> outW;
    std::vector<size_t> outSrc;
    outY.reserve(partY.size() + binY.size());
    outW.reserve(partW.size() + binW.size());
    outSrc.reserve(partSrc.size() + binY.size());

    size_t i = 0;
    size_t j = 0;
    while (i < partY.size() && j < binY.size()) {
      if (partY[i] <= binY[j]) {
        outY.push_back(partY[i]);
        outW.push_back(partW[i]);
        outSrc.push_back(partSrc[i]);
        ++i;
      } else {
        outY.push_back(binY[j]);
        outW.push_back(binW[j]);
        outSrc.push_back(leaf);
        ++j;
      }
    }
    while (i < partY.size()) {
      outY.push_back(partY[i]);
      outW.push_back(partW[i]);
      outSrc.push_back(partSrc[i]);
      ++i;
    }
    while (j < binY.size()) {
      outY.push_back(binY[j]);
      outW.push_back(binW[j]);
      outSrc.push_back(leaf);
      ++j;
    }
    partY.swap(outY);
    partW.swap(outW);
    partSrc.swap(outSrc);
  }
}

void AbsoluteErrorBranchAssignment::filterLeafFromPartition(size_t leaf,
                                                            size_t partition) {
  for (size_t o = 0; o < nOutputs_; ++o) {
    auto &partY = partYs_[partition][o];
    auto &partW = partWs_[partition][o];
    auto &partSrc = partSrcBin_[partition][o];
    if (partY.empty())
      continue;

    std::vector<float> outY;
    std::vector<float> outW;
    std::vector<size_t> outSrc;
    outY.reserve(partY.size());
    outW.reserve(partW.size());
    outSrc.reserve(partSrc.size());
    for (size_t i = 0; i < partY.size(); ++i) {
      if (partSrc[i] == leaf)
        continue;
      outY.push_back(partY[i]);
      outW.push_back(partW[i]);
      outSrc.push_back(partSrc[i]);
    }
    partY.swap(outY);
    partW.swap(outW);
    partSrc.swap(outSrc);
  }
}

double AbsoluteErrorBranchAssignment::objective() {
  return sumNumberOfSamples_ > 0.0
             ? weightedSumLoss_ / static_cast<double>(sumNumberOfSamples_)
             : 0.0;
}

void AbsoluteErrorBranchAssignment::addLeaf(size_t leaf, size_t partition) {
  weightedSumLoss_ -= partitionWeight_[partition] * partitionLoss_[partition];

  partitionWeight_[partition] += leafWeights_[leaf];
  partitionSampleCount_[partition] += leafSampleCounts_[leaf];
  sumNumberOfSamples_ += leafWeights_[leaf];

  mergeLeafIntoPartition(leaf, partition);

  partitionLoss_[partition] = computePartitionMae(partition);
  weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];

  assignments[leaf] = partition;
}

void AbsoluteErrorBranchAssignment::removeLeaf(size_t leaf) {
  const size_t partition = assignments[leaf];
  if (partition >= numPartitions)
    throw std::runtime_error(
        "removeLeaf: leaf is not assigned to a valid partition");

  weightedSumLoss_ -= partitionWeight_[partition] * partitionLoss_[partition];
  sumNumberOfSamples_ -= leafWeights_[leaf];
  partitionWeight_[partition] -= leafWeights_[leaf];
  partitionSampleCount_[partition] -= leafSampleCounts_[leaf];

  filterLeafFromPartition(leaf, partition);

  partitionLoss_[partition] = computePartitionMae(partition);
  weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];

  assignments[leaf] = absolute_error_branch::unassignedPartition(numPartitions);
}

double
AbsoluteErrorBranchAssignment::computePartitionMae(size_t partition) const {
  double total = 0.0;
  for (size_t o = 0; o < nOutputs_; ++o)
    total += Criterion::absoluteErrorPresorted(partYs_[partition][o],
                                               partWs_[partition][o])
                 .mae;
  return total;
}
