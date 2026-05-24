/**
 * @file SquaredErrorSplitter.cpp
 * @brief MSE splitter: root/children construction and interval scoring.
 */

#include "SquaredErrorSplitter.h"
#include "Criterion.h"

SplitCandidate SquaredErrorSplitter::makeRoot() {
  auto stats = makeEmptyStats();
  for (size_t idx = 0; idx < targets.n_cols; idx++) {
    const float v = targets(0, idx);
    const float w = sampleWeights(idx);
    stats[0] += w * v;
    stats[1] += w * v * v;
  }

  splitStats[0][targets.n_cols - 1] = stats;

  SplitCandidate root{.height = 0,
                      .start = 0,
                      .end = targets.n_cols - 1,
                      .score = score(stats, 0, targets.n_cols - 1),
                      .routingThreshold =
                          std::numeric_limits<double>::infinity()};
  fillIntervalMeta(root);
  return root;
}

float SquaredErrorSplitter::predict(const SplitCandidate &split) {
  const double W = split.nodeWeight;
  if (W <= 0.0)
    throw std::runtime_error(
        "Not possible to have a partition of weight 0 for a decision tree");

  return getStats(split)[0] / static_cast<float>(W);
}

double SquaredErrorSplitter::score(const std::vector<float> &stats, size_t l,
                                   size_t r) {
  return Criterion::squaredError(stats, intervalWeight(l, r));
}
void SquaredErrorSplitter::moveSample(std::vector<float> &rightStats,
                                      std::vector<float> &leftStats,
                                      size_t idx) {
  const float v = targets(0, idx);
  const float w = sampleWeights(idx);
  rightStats[0] -= w * v;
  leftStats[0] += w * v;
  rightStats[1] -= w * v * v;
  leftStats[1] += w * v * v;
}
