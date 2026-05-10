#pragma once

/**
 * @file GainHessianSplitter.h
 * @brief Gradient-boosting style splitter using first/second moment stats and ``Criterion::gainAndHessian``.
 */

#include "Splitter.h"

class GainHessianSplitter : public Splitter<float> {
public:
  GainHessianSplitter(arma::frowvec &X, arma::Mat<float> &y, double lambda)
      : Splitter(X, y, 2), lambda(lambda) {
    if (y.n_rows < 2)
      throw std::invalid_argument(
          "GainHessianSplitter requires y with at least 2 rows (gain, hessian)");
  }

  SplitCandidate makeRoot() override;
  float predict(const SplitCandidate &split) override;

  double score(const SplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }
  double score(const std::vector<float> &stats, size_t l, size_t r) override;

protected:
  void moveSample(std::vector<float> &rightStats, std::vector<float> &leftStats,
                  size_t idx) override;

private:
  double lambda;
};
