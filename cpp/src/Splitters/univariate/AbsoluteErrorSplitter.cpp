/**
 * @file AbsoluteErrorSplitter.cpp
 * @brief MAE splitter using wavelet-tree median queries on contiguous sample ranges.
 */

#include <cstddef>
#include "AbsoluteErrorSplitter.h"

#include <limits>

AbsoluteErrorSplitter::AbsoluteErrorSplitter(arma::frowvec &X,
                                             arma::frowvec &sampleWeights,
                                             arma::Mat<float> &y)
    : Splitter(X, sampleWeights, static_cast<size_t>(y.n_rows)), targets(y),
      nOutputs_(static_cast<size_t>(y.n_rows)),
      empty_stats_cache_(nOutputs_) {
  waveletTrees_.reserve(nOutputs_);
  for (size_t o = 0; o < nOutputs_; ++o)
    waveletTrees_.push_back(std::make_unique<WaveletTreeMAE>(
        arma::Row<float>(y.row(o)), arma::Row<float>(sampleWeights)));
}

std::vector<std::vector<double>> AbsoluteErrorSplitter::makeEmptyStats() {
  return std::vector<std::vector<double>>(nOutputs_);
}

UnivariateSplitCandidate AbsoluteErrorSplitter::makeRoot() {
  UnivariateSplitCandidate root{.height = 0,
                      .start = 0,
                      .end = targets.n_cols - 1,
                      .score = score(makeEmptyStats(), 0, targets.n_cols - 1),
                      .routingThreshold =
                          std::numeric_limits<double>::infinity()};
  fillIntervalMeta(root);
  return root;
}

std::vector<float>
AbsoluteErrorSplitter::predict(const UnivariateSplitCandidate &split) {
  std::vector<float> medians(nOutputs_, 0.f);
  for (size_t o = 0; o < nOutputs_; ++o)
    medians[o] = waveletTrees_[o]
                     ->quantileStatsForMedian(static_cast<int>(split.start),
                                              static_cast<int>(split.end))
                     .median_val;
  return medians;
}

double AbsoluteErrorSplitter::score(const std::vector<std::vector<double>> &stats,
                                    size_t l, size_t r) {
  (void)stats;
  double total = 0.0;
  for (size_t o = 0; o < nOutputs_; ++o)
    total += waveletTrees_[o]
                 ->quantileStatsForMedian(static_cast<int>(l),
                                          static_cast<int>(r))
                 .mae();
  return total;
}

double AbsoluteErrorSplitter::score(const UnivariateSplitCandidate &split) {
  return score(makeEmptyStats(), split.start, split.end);
}

const std::vector<std::vector<double>> &
AbsoluteErrorSplitter::getStats(const UnivariateSplitCandidate &split) {
  (void)split;
  return empty_stats_cache_;
}

void AbsoluteErrorSplitter::moveSample(std::vector<std::vector<double>> &rightStats,
                                       std::vector<std::vector<double>> &leftStats,
                                       size_t idx) {
  (void)rightStats;
  (void)leftStats;
  (void)idx;
}
