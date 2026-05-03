#include "SquaredErrorSplitter.h"

SplitCandidate SquaredErrorSplitter::makeRoot() {
  auto stats = makeEmptyStats();
  for (size_t idx = 0; idx < y.n_cols; idx++) {
    stats[0] += y(idx);
    stats[1] += y(idx) * y(idx);
  }

  splitStats[0][y.n_cols - 1] = stats;

  return {.height = 0,
          .start = 0,
          .end = y.n_cols - 1,
          .score = score(stats, y.n_cols),
          .routingThreshold = std::numeric_limits<double>::infinity()};
}

float SquaredErrorSplitter::predict(const SplitCandidate &split) {
  const size_t N = split.end - split.start + 1;
  if (N == 0)
    throw std::runtime_error(
        "Not possible to have a partition of size 0 for a decision tree");

  return getStats(split)[0] / static_cast<float>(N);
}

double SquaredErrorSplitter::score(const std::vector<float> &stats, size_t N) {
  if (N == 0)
    return 0;
  double ySum = stats[0], ySqrdSum = stats[1];

  double mean = ySum / N;

  return ySqrdSum - 2 * mean * ySum + N * mean * mean;
}

void SquaredErrorSplitter::moveSample(std::vector<float> &rightStats,
                                      std::vector<float> &leftStats, float v) {
  rightStats[0] -= v;
  leftStats[0] += v;
  rightStats[1] -= v * v;
  leftStats[1] += v * v;
}