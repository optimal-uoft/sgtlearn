/**
 * @file GainHessianSplitter.cpp
 * @brief Gradient/hessian splitter scoring via ``Criterion::gainAndHessian``.
 */

#include "GainHessianSplitter.h"

#include "Criterion.h"

SplitCandidate GainHessianSplitter::makeRoot() {
  auto stats = makeEmptyStats();
  for (size_t idx = 0; idx < derivatives.n_cols; ++idx) {
    const float w = sampleWeights(idx);
    stats[0] += w * derivatives(0, idx);
    stats[1] += w * derivatives(1, idx);
  }

  splitStats[0][derivatives.n_cols - 1] = stats;
  SplitCandidate root{.height = 0,
                      .start = 0,
                      .end = derivatives.n_cols - 1,
                      .score = score(stats, 0, derivatives.n_cols - 1),
                      .routingThreshold =
                          std::numeric_limits<double>::infinity()};
  fillIntervalMeta(root);
  return root;
}

float GainHessianSplitter::predict(const SplitCandidate &split) {
  const auto &stats = getStats(split);
  return static_cast<float>(score(stats, split.start, split.end));
}

double GainHessianSplitter::score(const std::vector<float> &stats, size_t l,
                                  size_t r) {
  return Criterion::gainAndHessian(stats, lambda);
}

void GainHessianSplitter::moveSample(std::vector<float> &rightStats,
                                     std::vector<float> &leftStats,
                                     size_t idx) {
  const float w = sampleWeights(idx);
  const float g = derivatives(0, idx);
  const float h = derivatives(1, idx);
  rightStats[0] -= w * g;
  leftStats[0] += w * g;
  rightStats[1] -= w * h;
  leftStats[1] += w * h;
}
