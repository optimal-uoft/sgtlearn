#pragma once

/**
 * @file SquaredErrorSplitter.h
 * @brief Regression splitter minimizing MSE via per-interval sums of ``y`` and ``y^2``.
 */

#include "Splitter.h"

class SquaredErrorSplitter : public Splitter<float> {
public:
  ~SquaredErrorSplitter() = default;
  SquaredErrorSplitter(arma::frowvec &X, arma::Mat<float> &y)
      : Splitter(X, y, 2) {}
  SplitCandidate makeRoot() override;
  float predict(const SplitCandidate &split) override;
  double score(const SplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }

  double score(const std::vector<float> &stats, size_t l, size_t r) override;

protected:
  void moveSample(std::vector<float> &rightStats, std::vector<float> &leftStats,
                  size_t idx) override;

};