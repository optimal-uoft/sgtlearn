/**
 * @file algorithms/TAO/ClassificationTaoAdapter.cpp
 */

#include "algorithms/TAO/ClassificationTaoAdapter.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tao {

ClassificationTaoAdapter::ClassificationTaoAdapter(
    ClassificationShapeGeneralizedTree &tree, const arma::fmat &X,
    const arma::Mat<size_t> &y, const arma::Row<float> &sampleWeights)
    : TaoAdapter(tree, X, sampleWeights), classificationTree_(tree), y_(y),
      classesPerOutput_(tree.classesPerOutput()),
      nOutputs_(static_cast<size_t>(y.n_rows)) {
  if (classesPerOutput_.size() != nOutputs_)
    throw std::invalid_argument(
        "ClassificationTaoAdapter: y.n_rows must match tree nOutputs");
}

LearningCriterion ClassificationTaoAdapter::routerCriterion() const {
  return classificationTree_.criterion();
}

void ClassificationTaoAdapter::childRewards(
    const std::vector<size_t> &childLeaves, arma::uword col,
    std::vector<double> &reward) const {
  reward.resize(childLeaves.size());

  if (nOutputs_ > 1) {
    const double inv = 1.0 / static_cast<double>(nOutputs_);
    for (size_t c = 0; c < childLeaves.size(); ++c) {
      const auto &hist = classificationTree_.classCounts[childLeaves[c]];
      size_t correct = 0;
      for (size_t o = 0; o < nOutputs_; ++o) {
        if (argMax(hist[o]) == y_(static_cast<arma::uword>(o), col))
          ++correct;
      }
      reward[c] = static_cast<double>(correct) * inv;
    }
    return;
  }

  const size_t label = y_(0, col);
  for (size_t c = 0; c < childLeaves.size(); ++c) {
    const auto &hist = classificationTree_.classCounts[childLeaves[c]];
    reward[c] = argMax(hist[0]) == label ? 1.0 : 0.0;
  }
}

void ClassificationTaoAdapter::recomputeLeafStats(
    const std::vector<std::vector<arma::uword>> &nodeSamples) {
  for (size_t ni = 0; ni < nodeSamples.size(); ++ni) {
    std::vector<std::vector<double>> counts(nOutputs_);
    for (size_t o = 0; o < nOutputs_; ++o)
      counts[o].assign(classesPerOutput_[o], 0.0);
    for (arma::uword col : nodeSamples[ni]) {
      const double w = static_cast<double>(w_(col));
      for (size_t o = 0; o < nOutputs_; ++o)
        counts[o][y_(static_cast<arma::uword>(o), col)] += w;
    }
    classificationTree_.classCounts[ni] = std::move(counts);
  }
}

void ClassificationTaoAdapter::refreshNodeBinMetadata(
    ShapeFunctionNode &node, const std::vector<arma::uword> &samples) {
  arma::Row<size_t> bins;
  initBinMetadata(node, samples, bins, "ClassificationTaoAdapter");
  const size_t nBins = node.binToPartition.size();
  node.splitClassCounts.assign(nBins,
                               std::vector<std::vector<double>>(nOutputs_));
  for (size_t b = 0; b < nBins; ++b)
    for (size_t o = 0; o < nOutputs_; ++o)
      node.splitClassCounts[b][o].assign(classesPerOutput_[o], 0.0);
  for (arma::uword col : samples) {
    const size_t bin = bins(col);
    const double weight = static_cast<double>(w_(col));
    for (size_t o = 0; o < nOutputs_; ++o)
      node.splitClassCounts[bin][o][y_(static_cast<arma::uword>(o), col)] +=
          weight;
  }
  node.splitLeafStats.clear();
}

} // namespace tao
