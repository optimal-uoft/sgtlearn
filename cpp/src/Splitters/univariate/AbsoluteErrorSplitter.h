#pragma once

/**
 * @file AbsoluteErrorSplitter.h
 * @brief MAE splitter using wavelet-tree median queries on contiguous sample ranges.
 */

#include <cstddef>
#include <memory>
#include <vector>
#include "Splitter.h"
#include "algorithms/WaveletTreeMAE.h"

/**
 * Multi-output MAE splitter: one ``WaveletTreeMAE`` per output row of ``y``.
 * The interval score is the SUM of per-output MAE and ``predict`` returns the
 * vector of per-output medians. Leaf stats are unused (empty per-output rows).
 */
class AbsoluteErrorSplitter : public Splitter<std::vector<double>, std::vector<float>> {
public:
  AbsoluteErrorSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                        arma::Mat<float> &y);
  UnivariateSplitCandidate makeRoot() override;
  std::vector<float> predict(const UnivariateSplitCandidate &split) override;
  double score(const UnivariateSplitCandidate &split) override;
  double score(const std::vector<std::vector<double>> &stats, size_t l,
               size_t r) override;
  const std::vector<std::vector<double>> &
  getStats(const UnivariateSplitCandidate &split) override;

protected:
  const arma::Mat<float> &targets;
  size_t nOutputs_;
  std::vector<std::unique_ptr<WaveletTreeMAE>> waveletTrees_;
  std::vector<std::vector<double>> empty_stats_cache_;

  std::vector<std::vector<double>> makeEmptyStats() override;

  void moveSample(std::vector<std::vector<double>> &rightStats,
                  std::vector<std::vector<double>> &leftStats,
                  size_t idx) override;
};
