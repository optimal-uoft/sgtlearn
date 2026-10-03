/**
 * @file algorithms/TAO/TaoObjective.cpp
 */

#include <cstddef>
#include "algorithms/TAO/TaoObjective.h"

#include "algorithms/missing_values.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

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

double TaoObjective::rewardSumForBins(
    const arma::Row<size_t> &bins,
    const std::vector<size_t> &binToPartition) const {
  double rewardSum = 0.0;
  for (size_t i = 0; i < nCare_; ++i) {
    const size_t bin = bins(care_.careCols[i]);
    if (bin >= binToPartition.size())
      throw std::runtime_error(
          "TaoObjective::rewardSumForBins: bin out of range");
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

  // Bin count includes a univariate discretizer's trailing NaN bin.
  const size_t nBins = disc.leafStats().size();
  if (nBins == 0)
    throw std::runtime_error(
        "TaoObjective::scoreDiscretizer: discretizer has no leaf stats");

  arma::Row<size_t> bins;
  disc.transform(X_, bins);

  // Relabel each bin by the true care reward, not the fitted histogram: the
  // router is fit on split weights, whose argmax can miss the best child.
  const size_t k = nCare_ == 0 ? 0 : care_.careRewards.front().size();
  std::vector<std::vector<double>> binReward(nBins,
                                             std::vector<double>(k, 0.0));
  std::vector<double> binWeight(nBins, 0.0);
  for (size_t i = 0; i < nCare_; ++i) {
    const size_t bin = bins(care_.careCols[i]);
    if (bin >= nBins)
      throw std::runtime_error(
          "TaoObjective::scoreDiscretizer: bin out of range");
    const double w = careWeight(i);
    binWeight[bin] += w;
    for (size_t c = 0; c < k; ++c)
      binReward[bin][c] += w * care_.careRewards[i][c];
  }
  binToPartitionOut.resize(nBins);
  for (size_t b = 0; b < nBins; ++b)
    binToPartitionOut[b] =
        binWeight[b] > 0.0 ? argMax(binReward[b]) : care_.dummyChild;

  return penalizedScore(rewardSumForBins(bins, binToPartitionOut),
                        complexityScale);
}

double TaoObjective::scoreBinAssignment(
    const arma::Row<size_t> &bins, const std::vector<size_t> &binToPartition,
    double complexityScale) const {
  return penalizedScore(rewardSumForBins(bins, binToPartition),
                        complexityScale);
}

std::optional<ThresholdCut>
TaoObjective::bestThresholdCut(size_t feature, size_t minLeafSize) const {
  if (nCare_ == 0)
    return std::nullopt;
  if (care_.careRewards.front().size() != 2)
    throw std::invalid_argument(
        "TaoObjective::bestThresholdCut: node must have exactly two children");

  std::vector<std::pair<float, size_t>> finite; // (value, care index)
  finite.reserve(nCare_);
  double total0 = 0.0, total1 = 0.0;
  double nan0 = 0.0, nan1 = 0.0, nanWeight = 0.0;
  for (size_t i = 0; i < nCare_; ++i) {
    const float value =
        X_(static_cast<arma::uword>(feature), care_.careCols[i]);
    const double w = careWeight(i);
    const double r0 = w * care_.careRewards[i][0];
    const double r1 = w * care_.careRewards[i][1];
    if (missing_values::is_finite(value)) {
      finite.emplace_back(value, i);
      total0 += r0;
      total1 += r1;
    } else {
      nan0 += r0;
      nan1 += r1;
      nanWeight += w;
    }
  }

  const size_t minSide = std::max<size_t>(minLeafSize, 1);
  const size_t n = finite.size();
  if (n < 2 * minSide)
    return std::nullopt;
  std::sort(finite.begin(), finite.end());

  const size_t nanChild =
      nanWeight > 0.0 ? (nan1 > nan0 ? 1 : 0) : care_.dummyChild;
  const double nanReward = nanChild == 0 ? nan0 : nan1;

  std::optional<ThresholdCut> best;
  double left0 = 0.0, left1 = 0.0;
  for (size_t j = 1; j < n; ++j) {
    const auto [prevValue, prevIdx] = finite[j - 1];
    const double w = careWeight(prevIdx);
    left0 += w * care_.careRewards[prevIdx][0];
    left1 += w * care_.careRewards[prevIdx][1];
    if (j < minSide)
      continue;
    if (n - j < minSide)
      break;
    // Same tie rule as Splitter::findBestSplit, so a depth-1 discretizer
    // trained on the two sides reproduces this cut.
    if (finite[j].first <= prevValue + static_cast<float>(1e-7))
      continue;
    const double leftTo0 = left0 + (total1 - left1);
    const double leftTo1 = left1 + (total0 - left0);
    const size_t leftChild = leftTo1 > leftTo0 ? 1 : 0;
    const double reward = std::max(leftTo0, leftTo1) + nanReward;
    if (!best || reward > best->rewardSum)
      best = ThresholdCut{prevValue, leftChild, nanChild, reward};
  }
  return best;
}

} // namespace tao
