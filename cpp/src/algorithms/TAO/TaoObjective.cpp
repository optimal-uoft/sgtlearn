/**
 * @file algorithms/TAO/TaoObjective.cpp
 */

#include <cstddef>
#include "algorithms/TAO/TaoObjective.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tao {

TaoObjective::TaoObjective(const NodeCareSet &care, const arma::fmat &X,
                           double lambda, double nodeSampleCount)
    : care_(care), X_(X), lambda_(lambda),
      nodeSampleCount_(nodeSampleCount), nCare_(care.size()),
      totalCareWeight_(0.0) {
  if (care_.careWeights.empty()) {
    totalCareWeight_ = static_cast<double>(nCare_);
    return;
  }
  for (double w : care_.careWeights)
    totalCareWeight_ += w;
  if (totalCareWeight_ <= 0.0)
    totalCareWeight_ = static_cast<double>(nCare_);
}

double TaoObjective::careWeight(size_t i) const {
  if (care_.careWeights.empty())
    return 1.0;
  return care_.careWeights[i];
}

size_t TaoObjective::argMax(const std::vector<double> &counts) {
  return static_cast<size_t>(std::distance(
      counts.begin(), std::max_element(counts.begin(), counts.end())));
}

double TaoObjective::meanReward(double rewardSum) const {
  return rewardSum / totalCareWeight_;
}

double TaoObjective::penalizedScore(double rewardSum,
                                    double complexityScale) const {
  if (lambda_ == 0.0 || complexityScale == 0.0 || totalCareWeight_ <= 0.0)
    return meanReward(rewardSum);
  return meanReward(rewardSum) -
         complexityScale * lambda_ * nodeSampleCount_ / totalCareWeight_;
}

double TaoObjective::rewardSumForDiscretizer(
    const ClassificationDiscretizer &disc,
    const std::vector<size_t> &binToPartition) const {
  arma::Row<size_t> bins;
  disc.transform(X_, bins);
  double rewardSum = 0.0;
  for (size_t i = 0; i < nCare_; ++i) {
    const size_t bin = bins(care_.careCols[i]);
    if (bin >= binToPartition.size())
      throw std::runtime_error(
          "TaoObjective::rewardSumForDiscretizer: bin out of range");
    const size_t child = binToPartition[bin];
    rewardSum += careWeight(i) * care_.careRewards[i][child];
  }
  return rewardSum;
}

double TaoObjective::scoreCurrent(const ShapeFunctionNode &node,
                                  double complexityScale) const {
  double rewardSum = 0.0;
  for (size_t i = 0; i < nCare_; ++i) {
    const size_t child = node.routeSampleToPartition(
        X_, static_cast<arma::uword>(care_.careCols[i]));
    rewardSum += careWeight(i) * care_.careRewards[i][child];
  }
  return penalizedScore(rewardSum, complexityScale);
}

double TaoObjective::scoreDummy() const {
  double rewardSum = 0.0;
  for (size_t i = 0; i < nCare_; ++i)
    rewardSum += careWeight(i) * care_.careRewards[i][care_.dummyChild];
  return meanReward(rewardSum);
}

double TaoObjective::scoreDiscretizer(
    ClassificationDiscretizer &disc,
    std::vector<size_t> &binToPartitionOut,
    double complexityScale) const {
  if (disc.numLeaves() < 1)
    return -std::numeric_limits<double>::infinity();

  const std::vector<std::vector<std::vector<double>>> &leafStats =
      disc.leafStats();
  if (leafStats.empty())
    throw std::runtime_error(
        "TaoObjective::scoreDiscretizer: discretizer has no leaf stats");

  binToPartitionOut.resize(leafStats.size());

  // TAO trains the inner discretizer on scalar care-set pseudo-labels (child
  // partition indices 0..k-1), so each bin's stats are [1][k] not multi-output.
  for (size_t b = 0; b < leafStats.size(); ++b) {
    if (leafStats[b].size() != 1)
      throw std::runtime_error(
          "TaoObjective::scoreDiscretizer: bin " + std::to_string(b) +
          " expected 1 output row of class counts, got " +
          std::to_string(leafStats[b].size()));
    if (leafStats[b][0].empty())
      throw std::runtime_error(
          "TaoObjective::scoreDiscretizer: bin " + std::to_string(b) +
          " class histogram is empty");
    binToPartitionOut[b] = argMax(leafStats[b][0]);
  }

  return penalizedScore(
      rewardSumForDiscretizer(disc, binToPartitionOut), complexityScale);
}

} // namespace tao
