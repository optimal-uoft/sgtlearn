#pragma once

/**
 * @file AbsoluteErrorSplitter.h
 * @brief MAE splitter using wavelet-tree median queries on contiguous sample ranges.
 */

#include "Splitter.h"
#include "algorithms/WaveletTreeMAE.h"

class AbsoluteErrorSplitter : public Splitter<float, float> {
public:
  AbsoluteErrorSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                        arma::Mat<float> &y);
  SplitCandidate makeRoot() override;
  float predict(const SplitCandidate &split) override;
  double score(const SplitCandidate &split) override;
  double score(const std::vector<float> &stats, size_t l, size_t r) override;
  const std::vector<float> &getStats(const SplitCandidate &split) override;

protected:
  const arma::Mat<float> &targets;
  WaveletTreeMAE waveletTree;
  std::vector<float> empty_stats_cache_;

  QuantileStats getMedianQuantileStats(size_t l, size_t r);

  void moveSample(std::vector<float> &rightStats, std::vector<float> &leftStats,
                  size_t idx) override;
};
