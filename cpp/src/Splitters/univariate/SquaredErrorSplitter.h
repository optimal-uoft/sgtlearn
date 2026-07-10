#pragma once

/**
 * @file SquaredErrorSplitter.h
 * @brief Regression splitter minimizing MSE via per-interval weighted sums of ``y`` and ``y^2``.
 */

#include "Splitter.h"

class SquaredErrorSplitter : public Splitter<double, float> {
public:
  ~SquaredErrorSplitter() override = default;
  SquaredErrorSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                       arma::Mat<float> &y)
      : Splitter(X, sampleWeights, 2), targets(y) {}
  UnivariateSplitCandidate makeRoot() override;
  float predict(const UnivariateSplitCandidate &split) override;
  double score(const UnivariateSplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }

  double score(const std::vector<double> &stats, size_t l, size_t r) override;

protected:
  const arma::Mat<float> &targets;

  void moveSample(std::vector<double> &rightStats, std::vector<double> &leftStats,
                  size_t idx) override;
};
