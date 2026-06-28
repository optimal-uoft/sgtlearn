/**
 * @file algorithms/TAO/TaoObjective.cpp
 */

#include "algorithms/TAO/TaoObjective.h"

#include "algorithms/missing_values.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tao {

TaoObjective::TaoObjective(const NodeCareSet &care, const arma::fmat &X,
                           double lambda)
    : care_(care), X_(X), lambda_(lambda), nCare_(care.size()) {}

size_t TaoObjective::argMax(const std::vector<double> &counts) {
  return static_cast<size_t>(std::distance(
      counts.begin(), std::max_element(counts.begin(), counts.end())));
}

size_t TaoObjective::routeValue(float value, const std::vector<float> &thresholds,
                                const std::vector<size_t> &binToPartition,
                                size_t nanPartition) {
  if (binToPartition.empty())
    return nanPartition;
  if (!missing_values::is_finite(value))
    return nanPartition;
  const auto it =
      std::lower_bound(thresholds.begin(), thresholds.end(), value);
  size_t bin = static_cast<size_t>(it - thresholds.begin());
  if (bin >= binToPartition.size())
    bin = binToPartition.size() - 1;
  return binToPartition[bin];
}

double TaoObjective::meanReward(double rewardSum) const {
  return rewardSum / static_cast<double>(nCare_);
}

double TaoObjective::penalizedScore(double rewardSum) const {
  return meanReward(rewardSum) - lambda_;
}

double TaoObjective::rewardSumForPartition(
    size_t feature, const std::vector<float> &thresholds,
    const std::vector<size_t> &binToPartition) const {
  double rewardSum = 0.0;
  for (size_t i = 0; i < nCare_; ++i) {
    const float v = X_(feature, care_.careCols[i]);
    const size_t child =
        routeValue(v, thresholds, binToPartition, care_.dummyChild);
    rewardSum += care_.careRewards[i][child];
  }
  return rewardSum;
}

double TaoObjective::scoreCurrent(const ShapeFunctionNode &node) const {
  double rewardSum = 0.0;
  for (size_t i = 0; i < nCare_; ++i) {
    const float v = X_(node.routingFeature, care_.careCols[i]);
    const size_t child = node.routeFeatureValueToPartition(v);
    rewardSum += care_.careRewards[i][child];
  }
  return penalizedScore(rewardSum);
}

double TaoObjective::scoreDummy() const {
  double rewardSum = 0.0;
  for (size_t i = 0; i < nCare_; ++i)
    rewardSum += care_.careRewards[i][care_.dummyChild];
  return meanReward(rewardSum);
}

double TaoObjective::scoreRouting(size_t feature,
                                  const std::vector<float> &thresholds,
                                  const std::vector<size_t> &binToPartition)
    const {
  return penalizedScore(
      rewardSumForPartition(feature, thresholds, binToPartition));
}

double TaoObjective::scoreDiscretizer(
    size_t feature, ClassificationDiscretizer &disc,
    std::vector<float> &thresholdsOut,
    std::vector<size_t> &binToPartitionOut) const {
  if (disc.numLeaves() < 1)
    return -std::numeric_limits<double>::infinity();

  const std::vector<double> &thr = disc.thresholds();
  const std::vector<std::vector<double>> &leafStats = disc.leafStats();
  thresholdsOut.resize(thr.size());
  for (size_t b = 0; b < thr.size(); ++b)
    thresholdsOut[b] = static_cast<float>(thr[b]);
  binToPartitionOut.resize(leafStats.size());
  for (size_t b = 0; b < leafStats.size(); ++b)
    binToPartitionOut[b] = argMax(leafStats[b]);

  return scoreRouting(feature, thresholdsOut, binToPartitionOut);
}

} // namespace tao
