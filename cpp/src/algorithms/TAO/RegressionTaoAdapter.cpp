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
  std::vector<size_t> childLeaves(k);
  std::vector<double> reward(k);
  std::vector<size_t> pseudolabels;
  for (arma::uword col : samples) {
    for (size_t c = 0; c < k; ++c)
      childLeaves[c] = walkToLeaf(children[c], col);
    childRewards(childLeaves, col, reward);

    double best = reward[0];
    double worst = reward[0];
    size_t bestChild = 0;
    for (size_t c = 1; c < k; ++c) {
      best = std::max(best, reward[c]);
      worst = std::min(worst, reward[c]);
      if (reward[c] > reward[bestChild])
        bestChild = c;
    }
    const double tol = 1e-9 * (1.0 + std::fabs(best));
    if (best - worst <= tol)
      continue;

    care.careCols.push_back(col);
    care.careRewards.push_back(reward);
    care.careWeights.push_back(static_cast<double>(w_(col)));
    pseudolabels.push_back(bestChild);
  }

  const size_t nCare = care.size();
  care.Xexp.set_size(numFeatures(), nCare);
  care.yexp.set_size(nCare);
  care.wexp.set_size(nCare);
  std::vector<double> childCounts(k, 0.0);
  for (size_t i = 0; i < nCare; ++i) {
    const size_t child = pseudolabels[i];
    const float wi = w_(care.careCols[i]);
    care.Xexp.col(i) = X_.col(care.careCols[i]);
    care.yexp(i) = child;
    care.wexp(i) = wi;
    childCounts[child] += static_cast<double>(wi);
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
