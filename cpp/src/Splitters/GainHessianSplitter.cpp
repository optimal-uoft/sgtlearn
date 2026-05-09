#include "GainHessianSplitter.h"

SplitCandidate GainHessianSplitter::makeRoot() {
  auto stats = makeEmptyStats();
  for (size_t idx = 0; idx < y.n_cols; ++idx) {
    stats[0] += y(0, idx);
    stats[1] += y(1, idx);
  }

  splitStats[0][y.n_cols - 1] = stats;
  return {.height = 0,
          .start = 0,
          .end = y.n_cols - 1,
          .score = score(stats, 0, y.n_cols - 1),
          .routingThreshold = std::numeric_limits<double>::infinity()};
}

float GainHessianSplitter::predict(const SplitCandidate &split) {
  const auto &stats = getStats(split);
  return static_cast<float>(score(stats, split.start, split.end));
}

double GainHessianSplitter::score(const std::vector<float> &stats, size_t l,
                                  size_t r) {
  (void)l;
  (void)r;
  const double g = static_cast<double>(stats[0]);
  const double h = static_cast<double>(stats[1]);
  return g / (h + lambda);
}

void GainHessianSplitter::moveSample(std::vector<float> &rightStats,
                                     std::vector<float> &leftStats,
                                     size_t idx) {
  const float g = y(0, idx);
  const float h = y(1, idx);
  rightStats[0] -= g;
  leftStats[0] += g;
  rightStats[1] -= h;
  leftStats[1] += h;
}
