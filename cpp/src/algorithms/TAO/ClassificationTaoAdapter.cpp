/**
 * @file algorithms/TAO/ClassificationTaoAdapter.cpp
 */

#include <cstddef>
#include "algorithms/TAO/ClassificationTaoAdapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tao {
namespace {

// Argmax within the block [offset, offset + nClasses) of a concatenated
// histogram, returned relative to the block.
size_t argMaxInBlock(const std::vector<double> &counts, size_t offset,
                     size_t nClasses) {
  size_t best = 0;
  double bestVal = -1.0;
  for (size_t c = 0; c < nClasses; ++c) {
    const double v = (offset + c < counts.size()) ? counts[offset + c] : 0.0;
    if (v > bestVal) {
      bestVal = v;
      best = c;
    }
  }
  return best;
}

void recomputeClassCounts(
    std::vector<std::vector<double>> &classCounts,
    const std::vector<std::vector<arma::uword>> &nodeSamples,
    const arma::Mat<size_t> &y, const arma::Row<float> &sampleWeights,
    const std::vector<size_t> &classOffsets, size_t nOutputs,
    size_t totalClasses) {
  for (size_t ni = 0; ni < nodeSamples.size(); ++ni) {
    std::vector<double> counts(totalClasses, 0.0);
    for (arma::uword col : nodeSamples[ni]) {
      const double w = static_cast<double>(sampleWeights(col));
      for (size_t o = 0; o < nOutputs; ++o)
        counts[classOffsets[o] +
               y(static_cast<arma::uword>(o), col)] += w;
    }
    classCounts[ni] = std::move(counts);
  }
}

} // namespace

ClassificationTaoAdapter::ClassificationTaoAdapter(
    ClassificationShapeGeneralizedTree &tree, const arma::fmat &X,
    const arma::Mat<size_t> &y, const arma::Row<float> &sampleWeights)
    : ShapeGeneralizedTaoAdapter(tree, X, sampleWeights),
      classificationTree_(tree), y_(y),
      classesPerOutput_(tree.classesPerOutput()),
      nOutputs_(static_cast<size_t>(y.n_rows)), totalClasses_(0) {
  if (classesPerOutput_.size() != nOutputs_)
    throw std::invalid_argument(
        "ClassificationTaoAdapter: y.n_rows must match tree nOutputs");
  classOffsets_.assign(nOutputs_, 0);
  for (size_t o = 0; o < nOutputs_; ++o) {
    classOffsets_[o] = totalClasses_;
    totalClasses_ += classesPerOutput_[o];
  }
}

LearningCriterion ClassificationTaoAdapter::routerCriterion() const {
  return classificationTree_.criterion();
}

void ClassificationTaoAdapter::childRewards(
    const std::vector<size_t> &childLeaves, arma::uword col,
    std::vector<double> &reward) const {
  reward.resize(childLeaves.size());

  if (nOutputs_ > 1) {
    // Reward = fraction of outputs each child's leaf classifies correctly.
    const double inv = 1.0 / static_cast<double>(nOutputs_);
    for (size_t c = 0; c < childLeaves.size(); ++c) {
      const auto &counts = classificationTree_.classCounts[childLeaves[c]];
      size_t correct = 0;
      for (size_t o = 0; o < nOutputs_; ++o) {
        const size_t pred =
            argMaxInBlock(counts, classOffsets_[o], classesPerOutput_[o]);
        if (pred == y_(static_cast<arma::uword>(o), col))
          ++correct;
      }
      reward[c] = static_cast<double>(correct) * inv;
    }
    return;
  }

  // Single-output: 1/x split uniformly across correct children.
  const size_t label = y_(0, col);
  size_t numCorrect = 0;
  for (size_t c = 0; c < childLeaves.size(); ++c) {
    const bool correct =
        argMaxInBlock(classificationTree_.classCounts[childLeaves[c]], 0,
                      classesPerOutput_[0]) == label;
    if (correct)
      ++numCorrect;
    reward[c] = correct ? 1.0 : 0.0;
  }
  if (numCorrect == 0)
    return;
  const double correctReward = 1.0 / static_cast<double>(numCorrect);
  for (size_t c = 0; c < reward.size(); ++c) {
    if (reward[c] > 0.0)
      reward[c] = correctReward;
  }
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

    // Single-output: care about the best (correct) children only. Multi-output:
    // care about all but the worst, matching the regression adapter.
    std::vector<size_t> good;
    for (size_t c = 0; c < k; ++c) {
      const bool keep = (nOutputs_ > 1) ? (reward[c] - worst > tol)
                                        : (best - reward[c] <= tol);
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
                       classOffsets_, nOutputs_, totalClasses_);
}

} // namespace tao
