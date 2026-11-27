/**
 * @file LeafAggregationBranchAssignment.cpp
 * @brief Aggregated branch-assignment objective for entropy, Gini, and MSE.
 */

#include "LeafAggregationBranchAssignment.h"

#include "Criterion.h"

#include <cstddef>
#include <stdexcept>

namespace {

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

} // namespace

LeafAggregationBranchAssignment::LeafAggregationBranchAssignment(
    LearningCriterion criterion, std::vector<size_t> &assignments,
    size_t numPartitions, std::vector<std::vector<std::vector<double>>> &stats,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts, size_t statsDim)
    : BranchAssignment(assignments, numPartitions, leafSampleCounts),
      criterion_(criterion), stats_(stats), leafWeights_(leafWeights),
      statsDim_(statsDim) {

  if (criterion_ != LearningCriterion::Entropy &&
      criterion_ != LearningCriterion::Gini &&
      criterion_ != LearningCriterion::SquaredError)
    throw std::invalid_argument(
        "LeafAggregationBranchAssignment: requires Entropy, Gini, or SquaredError");

  if (assignments.size() != stats.size() || stats.size() != leafWeights.size())
    throw std::runtime_error("bin statistics must all be the same length");
  if (leafSampleCounts.size() != stats.size())
    throw std::runtime_error(
        "leafSampleCounts must have the same length as bin statistics");

  for (size_t b = 0; b < assignments.size(); ++b) {
    if (assignments[b] >= numPartitions)
      throw std::runtime_error("assignments[b] must be a valid partition index");
  }

  partitionStats_ = std::vector(
      numPartitions, std::vector<std::vector<double>>(statsDim_));
  partitionWeight_.assign(numPartitions, 0.0);
  partitionLoss_.assign(numPartitions, 0.0);

  const size_t numLeaves = assignments.size();
  for (size_t b = 0; b < numLeaves; ++b) {
    const size_t partition = assignments[b];
    partitionWeight_[partition] += leafWeights_[b];
    partitionSampleCount_[partition] += leafSampleCounts_[b];
    for (size_t d = 0; d < statsDim_; ++d)
      accumulateStat(partitionStats_[partition][d], stats_[b][d]);
  }

  for (size_t partition = 0; partition < numPartitions; ++partition) {
    sumNumberOfSamples_ += partitionWeight_[partition];
    partitionLoss_[partition] = computePartitionLoss(partition);
    weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];
  }
}

double LeafAggregationBranchAssignment::objective() {
  return sumNumberOfSamples_ > 0.0 ? weightedSumLoss_ / sumNumberOfSamples_
                                   : 0.0;
}

void LeafAggregationBranchAssignment::addLeaf(size_t leaf, size_t partition) {
  weightedSumLoss_ -= partitionWeight_[partition] * partitionLoss_[partition];

  for (size_t d = 0; d < statsDim_; ++d)
    accumulateStat(partitionStats_[partition][d], stats_[leaf][d]);
  partitionWeight_[partition] += leafWeights_[leaf];
  partitionSampleCount_[partition] += leafSampleCounts_[leaf];
  sumNumberOfSamples_ += leafWeights_[leaf];

  partitionLoss_[partition] = computePartitionLoss(partition);
  weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];

  assignments[leaf] = partition;
}

void LeafAggregationBranchAssignment::removeLeaf(size_t leaf) {
  const size_t partition = assignments[leaf];
  if (partition >= numPartitions)
    throw std::runtime_error(
        "removeLeaf: leaf is not assigned to a valid partition");

  weightedSumLoss_ -= partitionWeight_[partition] * partitionLoss_[partition];
  sumNumberOfSamples_ -= leafWeights_[leaf];
  partitionWeight_[partition] -= leafWeights_[leaf];
  partitionSampleCount_[partition] -= leafSampleCounts_[leaf];
  for (size_t d = 0; d < statsDim_; ++d)
    subtractStat(partitionStats_[partition][d], stats_[leaf][d]);

  partitionLoss_[partition] = computePartitionLoss(partition);
  weightedSumLoss_ += partitionWeight_[partition] * partitionLoss_[partition];

  assignments[leaf] = numPartitions;
}

double LeafAggregationBranchAssignment::computePartitionLoss(size_t i) const {
  switch (criterion_) {
  case LearningCriterion::Entropy:
    return Criterion::entropy(partitionStats_[i]);
  case LearningCriterion::Gini:
    return Criterion::gini(partitionStats_[i]);
  case LearningCriterion::SquaredError:
    return Criterion::squaredError(partitionStats_[i], partitionWeight_[i]);
  default:
    throw std::invalid_argument(
        "LeafAggregationBranchAssignment: unsupported learning criterion");
  }
}
