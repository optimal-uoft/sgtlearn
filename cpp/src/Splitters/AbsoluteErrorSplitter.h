#pragma once
#include "Splitter.h"
#include "algorithms/WaveletTreeMAE.h"

class AbsoluteErrorSplitter : public Splitter<float> {
  WaveletTreeMAE waveletTree;
  std::vector<float> empty_stats_cache_;

  QuantileStats getMedianQuantileStats(size_t l, size_t r);
public:
  AbsoluteErrorSplitter(arma::frowvec &X, arma::Mat<float> &y);
  SplitCandidate makeRoot() override;
  float predict(const SplitCandidate &split) override;
  double score(const std::vector<float> &stats, size_t l, size_t r) override;
  double score(const SplitCandidate &split) override;

protected:
  void moveSample(std::vector<float> &, std::vector<float> &, size_t) override {};

  const std::vector<float> &getStats(const SplitCandidate &split) override;
};