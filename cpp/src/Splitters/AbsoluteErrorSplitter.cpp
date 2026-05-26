/**
 * @file AbsoluteErrorSplitter.cpp
 * @brief MAE splitter using wavelet-tree median queries on contiguous sample ranges.
 */

#include "AbsoluteErrorSplitter.h"

#include <limits>

QuantileStats AbsoluteErrorSplitter::getMedianQuantileStats(size_t l, size_t r) {
  return waveletTree.quantileStatsForMedian(static_cast<int>(l),
                                            static_cast<int>(r));
}

AbsoluteErrorSplitter::AbsoluteErrorSplitter(arma::frowvec &X,
                                             arma::frowvec &sampleWeights,
                                             arma::Mat<float> &y)
    : Splitter(X, sampleWeights, 0), targets(y),
      waveletTree(arma::Row<float>(y.row(0)), arma::Row<float>(sampleWeights)) {}

SplitCandidate AbsoluteErrorSplitter::makeRoot() {
  SplitCandidate root{.height = 0,
                      .start = 0,
                      .end = targets.n_cols - 1,
                      .score = score(makeEmptyStats(), 0, targets.n_cols - 1),
                      .routingThreshold =
                          std::numeric_limits<double>::infinity()};
  fillIntervalMeta(root);
  return root;
}

float AbsoluteErrorSplitter::predict(const SplitCandidate &split) {
  return getMedianQuantileStats(split.start, split.end).median_val;
}

double AbsoluteErrorSplitter::score(const std::vector<double> &stats, size_t l,
                                    size_t r) {
  (void)stats;
  return getMedianQuantileStats(l, r).mae();
}

double AbsoluteErrorSplitter::score(const SplitCandidate &split) {
  return score(makeEmptyStats(), split.start, split.end);
}

const std::vector<double> &
AbsoluteErrorSplitter::getStats(const SplitCandidate &split) {
  (void)split;
  return empty_stats_cache_;
}

void AbsoluteErrorSplitter::moveSample(std::vector<double> &rightStats,
                                       std::vector<double> &leftStats,
                                       size_t idx) {
  (void)rightStats;
  (void)leftStats;
  (void)idx;
}
