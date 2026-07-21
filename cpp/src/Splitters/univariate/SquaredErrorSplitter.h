#pragma once

/**
 * @file SquaredErrorSplitter.h
 * @brief Regression splitter minimizing MSE via per-interval weighted sums of ``y`` and ``y^2``.
 */

#include <cstddef>
#include <vector>
#include "Splitter.h"

/**
 * Multi-output MSE splitter. ``y`` has shape ``(nOutputs, nSamples)``; stats are
 * nested ``[output][Σw·y, Σw·y²]`` and the score is the SUM of per-output
 * squared error. Single-output (one row) matches the old scalar path.
 */
class SquaredErrorSplitter : public Splitter<std::vector<double>, std::vector<float>> {
public:
  ~SquaredErrorSplitter() override = default;
  SquaredErrorSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                       arma::Mat<float> &y)
      : Splitter(X, sampleWeights, static_cast<size_t>(y.n_rows)),
        targets(y), nOutputs_(static_cast<size_t>(y.n_rows)) {}
  UnivariateSplitCandidate makeRoot() override;
  std::vector<float> predict(const UnivariateSplitCandidate &split) override;
  double score(const UnivariateSplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }

  double score(const std::vector<std::vector<double>> &stats, size_t l,
               size_t r) override;

protected:
  const arma::Mat<float> &targets;
  size_t nOutputs_;

  std::vector<std::vector<double>> makeEmptyStats() override;

  void moveSample(std::vector<std::vector<double>> &rightStats,
                  std::vector<std::vector<double>> &leftStats,
                  size_t idx) override;
};
