/**
 * @file algorithms/TAO/TaoAdapter.cpp
 * @brief Shared TAO adapter logic (tree walk, care-set construction, bin metadata).
 */

#include "algorithms/TAO/TaoAdapter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tao {

TaoAdapter::TaoAdapter(ShapeGeneralizedTree &tree, const arma::fmat &X,
                       const arma::Row<float> &sampleWeights)
    : tree_(tree), X_(X), w_(sampleWeights) {}

size_t TaoAdapter::argMax(const std::vector<double> &counts) {
  return static_cast<size_t>(std::distance(
      counts.begin(), std::max_element(counts.begin(), counts.end())));
}

size_t TaoAdapter::walkToLeaf(size_t startNode, arma::uword col) const {
  const auto &treeNodes = tree_.nodes();
  const auto &treeChildren = tree_.childIndices();
  size_t idx = startNode;
  while (!treeNodes[idx].isLeaf) {
    const size_t part = treeNodes[idx].routeSampleToPartition(X_, col);
    const auto &children = treeChildren[idx];
    if (part >= children.size())
      throw std::runtime_error("tao::walkToLeaf: routed partition out of range");
    idx = children[part];
  }
  return idx;
}

NodeCareSet TaoAdapter::buildCareSet(const std::vector<arma::uword> &samples,
                                     const std::vector<size_t> &children) const {
  NodeCareSet care;
  const size_t k = children.size();
  const bool onlyBest = careSetKeepsOnlyBest();
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
    for (size_t c = 0; c < k; ++c) {
      const bool keep =
          onlyBest ? (best - reward[c] <= tol) : (reward[c] - worst > tol);
      if (keep)
        good.push_back(c);
    }

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
    const float wRow =
        onlyBest ? wi / static_cast<float>(goodChildren[i].size()) : wi;
    for (size_t child : goodChildren[i]) {
      care.Xexp.col(pos) = X_.col(care.careCols[i]);
      care.yexp(pos) = child;
      care.wexp(pos) = wRow;
      childCounts[child] += static_cast<double>(wi);
      ++pos;
    }
  }
  care.dummyChild = argMax(childCounts);
  return care;
}

void TaoAdapter::initBinMetadata(ShapeFunctionNode &node,
                                 const std::vector<arma::uword> &samples,
                                 arma::Row<size_t> &binsOut,
                                 const char *caller) const {
  if (!node.innerDiscretizer)
    throw std::runtime_error(std::string(caller) +
                             ": accepted router has no discretizer");
  node.innerDiscretizer->transform(X_, binsOut);
  const size_t nBins = node.binToPartition.size();
  node.binSampleCounts.assign(nBins, 0);
  node.splitBinWeights.assign(nBins, 0.0);
  for (arma::uword col : samples) {
    const size_t bin = binsOut(col);
    if (bin >= nBins)
      throw std::runtime_error(std::string(caller) +
                               ": accepted router bin out of range");
    ++node.binSampleCounts[bin];
    node.splitBinWeights[bin] += static_cast<double>(w_(col));
  }
}

} // namespace tao
