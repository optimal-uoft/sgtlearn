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
 * ``[Σw·y0, Σw·y0², Σw·y1, Σw·y1², ...]`` (length ``2 * nOutputs``) and the score
 * is the SUM of per-output squared error. Single-output (one row) matches the
 * old scalar path.
 */
class SquaredErrorSplitter : public Splitter<double, std::vector<float>> {
public:
  ~SquaredErrorSplitter() override = default;
  SquaredErrorSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                       arma::Mat<float> &y)
      : Splitter(X, sampleWeights, 2 * static_cast<size_t>(y.n_rows)),
        targets(y), nOutputs_(static_cast<size_t>(y.n_rows)) {}
  UnivariateSplitCandidate makeRoot() override;
  std::vector<float> predict(const UnivariateSplitCandidate &split) override;
  double score(const UnivariateSplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }

  double score(const std::vector<double> &stats, size_t l, size_t r) override;

protected:
  const arma::Mat<float> &targets;
  size_t nOutputs_;

  void moveSample(std::vector<double> &rightStats, std::vector<double> &leftStats,
                  size_t idx) override;
};
