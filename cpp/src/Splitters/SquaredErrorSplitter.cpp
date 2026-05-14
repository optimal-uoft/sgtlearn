/**
 * @file SquaredErrorSplitter.cpp
 * @brief MSE splitter: root/children construction and interval scoring.
 */

#include "SquaredErrorSplitter.h"
#include "Criterion.h"

SplitCandidate SquaredErrorSplitter::makeRoot() {
  auto stats = makeEmptyStats();
  for (size_t idx = 0; idx < y.n_cols; idx++) {
    const float v = y(0, idx);
    stats[0] += v;
    stats[1] += v * v;
  }

  splitStats[0][y.n_cols - 1] = stats;

  return {.height = 0,
          .start = 0,
          .end = y.n_cols - 1,
          .score = score(stats, 0, y.n_cols - 1),
          .routingThreshold = std::numeric_limits<double>::infinity()};
}

float SquaredErrorSplitter::predict(const SplitCandidate &split) {
  const size_t N = split.end - split.start + 1;
  if (N == 0)
    throw std::runtime_error(
        "Not possible to have a partition of size 0 for a decision tree");

  return getStats(split)[0] / static_cast<float>(N);
}

double SquaredErrorSplitter::score(const std::vector<float> &stats, size_t l,
                                   size_t r) {
  return Criterion::squaredError(stats, r - l + 1);
}
void SquaredErrorSplitter::moveSample(std::vector<float> &rightStats,
                                      std::vector<float> &leftStats,
                                      size_t idx) {
  const float v = y(0, idx);
  rightStats[0] -= v;
  leftStats[0] += v;
  rightStats[1] -= v * v;
  leftStats[1] += v * v;
}