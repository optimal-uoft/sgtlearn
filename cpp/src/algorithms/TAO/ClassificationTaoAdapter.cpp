/**
 * @file algorithms/TAO/ClassificationTaoAdapter.cpp
 */

#include "algorithms/TAO/ClassificationTaoAdapter.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace tao {
namespace {

void recomputeClassCounts(
    std::vector<std::vector<double>> &classCounts,
    const std::vector<std::vector<arma::uword>> &nodeSamples,
    const arma::Row<size_t> &y, const arma::Row<float> &sampleWeights,
    size_t numClasses) {
  for (size_t ni = 0; ni < nodeSamples.size(); ++ni) {
    std::vector<double> counts(numClasses, 0.0);
    for (arma::uword col : nodeSamples[ni])
      counts[y(col)] += static_cast<double>(sampleWeights(col));
    classCounts[ni] = std::move(counts);
  }
}

} // namespace

ClassificationTaoAdapter::ClassificationTaoAdapter(
    ClassificationShapeGeneralizedTree &tree, const arma::fmat &X,
    const arma::Row<size_t> &y, const arma::Row<float> &sampleWeights)
    : ShapeGeneralizedTaoAdapter(tree, X, sampleWeights),
      classificationTree_(tree), y_(y) {}

LearningCriterion ClassificationTaoAdapter::routerCriterion() const {
  return classificationTree_.criterion();
}

void ClassificationTaoAdapter::childRewards(
    const std::vector<size_t> &childLeaves, arma::uword col,
    std::vector<double> &reward) const {
  const size_t label = y_(col);
  for (size_t c = 0; c < childLeaves.size(); ++c)
    reward[c] =
        (argMax(classificationTree_.classCounts[childLeaves[c]]) == label)
            ? 1.0
            : 0.0;
}

NodeCareSet ClassificationTaoAdapter::buildCareSet(
    const std::vector<arma::uword> &samples,
    const std::vector<size_t> &children) const {
  NodeCareSet care;
  const size_t k = children.size();
  std::vector<std::vector<size_t>> goodChildren;
  std::vector<size_t> childLeaves(k);
  std::vector<double> reward(k);
  for (arma::uword col : samples) {
    for (size_t c = 0; c < k; ++c)
      childLeaves[c] = walkToLeaf(children[c], col);
    childRewards(childLeaves, col, reward);

    double best = reward[0];
    double worst = reward[0];
    for (size_t c = 1; c < k; ++c) {
      best = std::max(best, reward[c]);
      worst = std::min(worst, reward[c]);
    }
    const double tol = 1e-9 * (1.0 + std::fabs(best));
    if (best - worst <= tol)
      continue;

    std::vector<size_t> good;
    for (size_t c = 0; c < k; ++c)
      if (best - reward[c] <= tol)
        good.push_back(c);

    care.careCols.push_back(col);
    care.careRewards.push_back(reward);
    care.careWeights.push_back(static_cast<double>(w_(col)));
    goodChildren.push_back(std::move(good));
  }

  size_t nExpanded = 0;
  for (const auto &good : goodChildren)
    nExpanded += good.size();
  care.Xexp.set_size(numFeatures(), nExpanded);
  care.yexp.set_size(nExpanded);
  care.wexp.set_size(nExpanded);
  std::vector<double> childCounts(k, 0.0);
  size_t pos = 0;
  for (size_t i = 0; i < care.size(); ++i) {
    const float wi = w_(care.careCols[i]);
    for (size_t child : goodChildren[i]) {
      care.Xexp.col(pos) = X_.col(care.careCols[i]);
      care.yexp(pos) = child;
      care.wexp(pos) = wi;
      childCounts[child] += static_cast<double>(wi);
      ++pos;
    }
  }
  care.dummyChild = argMax(childCounts);
  return care;
}

void ClassificationTaoAdapter::recomputeLeafStats(
    const std::vector<std::vector<arma::uword>> &nodeSamples) {
  recomputeClassCounts(classificationTree_.classCounts, nodeSamples, y_, w_,
                       classificationTree_.numClasses());
}

} // namespace tao
