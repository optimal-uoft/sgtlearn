#pragma once

/**
 * @file AbsoluteErrorSplitter.h
 * @brief MAE splitter using wavelet-tree median queries on contiguous sample ranges.
 */

#include "Splitter.h"
#include "algorithms/WaveletTreeMAE.h"

class AbsoluteErrorSplitter : public Splitter<double, float> {
public:
  AbsoluteErrorSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                        arma::Mat<float> &y);
  UnivariateSplitCandidate makeRoot() override;
  float predict(const UnivariateSplitCandidate &split) override;
  double score(const UnivariateSplitCandidate &split) override;
  double score(const std::vector<double> &stats, size_t l, size_t r) override;
  const std::vector<double> &getStats(const UnivariateSplitCandidate &split) override;

protected:
  const arma::Mat<float> &targets;
  WaveletTreeMAE waveletTree;
  std::vector<double> empty_stats_cache_;

  QuantileStats getMedianQuantileStats(size_t l, size_t r);

  void moveSample(std::vector<double> &rightStats, std::vector<double> &leftStats,
                  size_t idx) override;
};
