/**
 * @file algorithms/TAO/RegressionTaoAdapter.cpp
 */

#include "algorithms/TAO/RegressionTaoAdapter.h"

#include "Criterion.h"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace tao {

RegressionTaoAdapter::RegressionTaoAdapter(
    RegressionShapeGeneralizedTree &tree, const arma::fmat &X,
    const arma::Mat<float> &y, const arma::Row<float> &sampleWeights)
    : TaoAdapter(tree, X, sampleWeights), regressionTree_(tree), y_(y),
      squared_(tree.criterion() == LearningCriterion::SquaredError) {}

LearningCriterion RegressionTaoAdapter::routerCriterion() const {
  return LearningCriterion::Gini;
}

void RegressionTaoAdapter::childRewards(const std::vector<size_t> &childLeaves,
                                        arma::uword col,
                                        std::vector<double> &reward) const {
  const auto &leafPred = regressionTree_.leafPredictions();
  const size_t nOutputs = regressionTree_.nOutputs();
  for (size_t c = 0; c < childLeaves.size(); ++c) {
    const std::vector<double> &pred = leafPred[childLeaves[c]];
    double loss = 0.0;
    for (size_t o = 0; o < nOutputs; ++o) {
      const double target =
          static_cast<double>(y_(static_cast<arma::uword>(o), col));
      const double d = (o < pred.size() ? pred[o] : 0.0) - target;
      loss += squared_ ? (d * d) : std::fabs(d);
    }
    reward[c] = -loss;
  }
}

void RegressionTaoAdapter::recomputeLeafStats(
    const std::vector<std::vector<arma::uword>> &nodeSamples) {
  auto &treeNodes = regressionTree_.mutableNodes();
  auto &leafPred = regressionTree_.mutableLeafPredictions();
  const size_t nOutputs = regressionTree_.nOutputs();

  for (size_t ni = 0; ni < treeNodes.size(); ++ni) {
    if (!treeNodes[ni].isLeaf)
      continue;
    const std::vector<arma::uword> &cols = nodeSamples[ni];

    if (squared_) {
      std::vector<double> preds(nOutputs, 0.0);
      std::vector<float> stats(2 * nOutputs, 0.0f);
      double sumW = 0.0;
      for (arma::uword col : cols)
        sumW += static_cast<double>(w_(col));
      for (size_t o = 0; o < nOutputs; ++o) {
        double sumWY = 0.0;
        double sumWY2 = 0.0;
        for (arma::uword col : cols) {
          const double wi = static_cast<double>(w_(col));
          const double v =
              static_cast<double>(y_(static_cast<arma::uword>(o), col));
          sumWY += wi * v;
          sumWY2 += wi * v * v;
        }
        preds[o] = sumW > 0.0 ? sumWY / sumW : 0.0;
        stats[2 * o] = static_cast<float>(sumWY);
        stats[2 * o + 1] = static_cast<float>(sumWY2);
      }
      leafPred[ni] = std::move(preds);
      if (ni < regressionTree_.leafRegressionStats.size())
        regressionTree_.leafRegressionStats[ni] = std::move(stats);
    } else {
      std::vector<double> preds(nOutputs, 0.0);
      for (size_t o = 0; o < nOutputs; ++o) {
        std::vector<float> ys;
        std::vector<float> ws;
        ys.reserve(cols.size());
        ws.reserve(cols.size());
        for (arma::uword col : cols) {
          ys.push_back(y_(static_cast<arma::uword>(o), col));
          ws.push_back(w_(col));
        }
        preds[o] = Criterion::absoluteError(ys, ws).median;
      }
      leafPred[ni] = std::move(preds);
      if (ni < regressionTree_.leafRegressionStats.size())
        regressionTree_.leafRegressionStats[ni].clear();
    }
    if (ni < regressionTree_.leafNumSamples.size())
      regressionTree_.leafNumSamples[ni] = cols.size();
  }
}

void RegressionTaoAdapter::refreshNodeBinMetadata(
    ShapeFunctionNode &node, const std::vector<arma::uword> &samples) {
  arma::Row<size_t> bins;
  initBinMetadata(node, samples, bins, "RegressionTaoAdapter");
  const size_t nBins = node.binToPartition.size();
  node.splitClassCounts.clear();
  if (!squared_) {
    node.splitLeafStats.clear();
    return;
  }
  node.splitLeafStats.assign(
      nBins,
      std::vector<std::vector<double>>(y_.n_rows, std::vector<double>(2, 0.0)));
  for (arma::uword col : samples) {
    const size_t bin = bins(col);
    const double weight = static_cast<double>(w_(col));
    for (arma::uword o = 0; o < y_.n_rows; ++o) {
      const double value = static_cast<double>(y_(o, col));
      node.splitLeafStats[bin][o][0] += weight * value;
      node.splitLeafStats[bin][o][1] += weight * value * value;
    }
  }
}

} // namespace tao
