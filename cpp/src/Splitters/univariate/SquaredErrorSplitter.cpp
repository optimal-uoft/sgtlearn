/**
 * @file SquaredErrorSplitter.cpp
 * @brief MSE splitter: root/children construction and interval scoring.
 */

#include <stdexcept>
#include <limits>
#include <cstddef>
#include "SquaredErrorSplitter.h"
#include "Criterion.h"

std::vector<std::vector<double>> SquaredErrorSplitter::makeEmptyStats() {
  std::vector<std::vector<double>> stats(nOutputs_);
  for (size_t o = 0; o < nOutputs_; ++o)
    stats[o] = {0.0, 0.0};
  return stats;
}

UnivariateSplitCandidate SquaredErrorSplitter::makeRoot() {
  auto stats = makeEmptyStats();
  for (size_t idx = 0; idx < targets.n_cols; idx++) {
    const double w = static_cast<double>(sampleWeights(idx));
    for (size_t o = 0; o < nOutputs_; ++o) {
      const double v = static_cast<double>(targets(o, idx));
      stats[o][0] += w * v;
      stats[o][1] += w * v * v;
    }
  }

  splitStats[0][targets.n_cols - 1] = stats;

  UnivariateSplitCandidate root{.height = 0,
                      .start = 0,
                      .end = targets.n_cols - 1,
                      .score = score(stats, 0, targets.n_cols - 1),
                      .routingThreshold =
                          std::numeric_limits<double>::infinity()};
  fillIntervalMeta(root);
  return root;
}

std::vector<float>
SquaredErrorSplitter::predict(const UnivariateSplitCandidate &split) {
  const double W = split.nodeWeight;
  if (W <= 0.0)
    throw std::runtime_error(
        "Not possible to have a partition of weight 0 for a decision tree");
  const auto &stats = getStats(split);
  std::vector<float> means(nOutputs_, 0.f);
  for (size_t o = 0; o < nOutputs_; ++o)
    means[o] = static_cast<float>(stats[o][0] / W);
  return means;
}

double SquaredErrorSplitter::score(const std::vector<std::vector<double>> &stats,
                                   size_t l, size_t r) {
  return Criterion::squaredError(stats, intervalWeight(l, r));
}

void SquaredErrorSplitter::moveSample(std::vector<std::vector<double>> &rightStats,
                                      std::vector<std::vector<double>> &leftStats,
                                      size_t idx) {
  const double w = static_cast<double>(sampleWeights(idx));
  for (size_t o = 0; o < nOutputs_; ++o) {
    const double v = static_cast<double>(targets(o, idx));
    rightStats[o][0] -= w * v;
    leftStats[o][0] += w * v;
    rightStats[o][1] -= w * v * v;
    leftStats[o][1] += w * v * v;
  }
}
