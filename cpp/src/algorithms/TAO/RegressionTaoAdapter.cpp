/**
 * @file algorithms/TAO/RegressionTaoAdapter.cpp
 */

#include "algorithms/TAO/RegressionTaoAdapter.h"

#include "Criterion.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace tao {

RegressionTaoAdapter::RegressionTaoAdapter(
    RegressionShapeGeneralizedTree &tree, const arma::fmat &X,
    const arma::Row<float> &y, const arma::Row<float> &sampleWeights)
    : ShapeGeneralizedTaoAdapter(tree, X, sampleWeights), regressionTree_(tree),
      y_(y),
      squared_(tree.criterion() == LearningCriterion::SquaredError) {}

LearningCriterion RegressionTaoAdapter::routerCriterion() const {
  return LearningCriterion::Gini;
}

void RegressionTaoAdapter::childRewards(
    const std::vector<size_t> &childLeaves, arma::uword col,
    std::vector<double> &reward) const {
  const auto &leafPred = regressionTree_.leafPredictions();
  const double target = static_cast<double>(y_(col));
  for (size_t c = 0; c < childLeaves.size(); ++c) {
    const double d = leafPred[childLeaves[c]] - target;
    reward[c] = squared_ ? -(d * d) : -std::fabs(d);
  }
}

NodeCareSet RegressionTaoAdapter::buildCareSet(
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
      if (reward[c] - worst > tol)
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

void RegressionTaoAdapter::recomputeLeafStats(
    const std::vector<std::vector<arma::uword>> &nodeSamples) {
  auto &nodes = regressionTree_.mutableNodes();
  auto &leafPred = regressionTree_.mutableLeafPredictions();

  for (size_t ni = 0; ni < nodes.size(); ++ni) {
    if (!nodes[ni].isLeaf)
      continue;
    const std::vector<arma::uword> &cols = nodeSamples[ni];

    if (squared_) {
      double sumWY = 0.0;
      double sumWY2 = 0.0;
      double sumW = 0.0;
      for (arma::uword col : cols) {
        const double wi = static_cast<double>(w_(col));
        const double v = static_cast<double>(y_(col));
        sumWY += wi * v;
        sumWY2 += wi * v * v;
        sumW += wi;
      }
      leafPred[ni] = sumW > 0.0 ? sumWY / sumW : 0.0;
      if (ni < regressionTree_.leafRegressionStats.size())
        regressionTree_.leafRegressionStats[ni] = {static_cast<float>(sumWY),
                                                 static_cast<float>(sumWY2)};
    } else {
      std::vector<float> ys;
      std::vector<float> ws;
      ys.reserve(cols.size());
      ws.reserve(cols.size());
      for (arma::uword col : cols) {
        ys.push_back(y_(col));
        ws.push_back(w_(col));
      }
      leafPred[ni] = Criterion::absoluteError(ys, ws).median;
      if (ni < regressionTree_.leafRegressionStats.size())
        regressionTree_.leafRegressionStats[ni].clear();
    }
    if (ni < regressionTree_.leafNumSamples.size())
      regressionTree_.leafNumSamples[ni] = cols.size();
  }
}

} // namespace tao
