#pragma once

/**
 * @file GainHessianSplitter.h
 * @brief Gradient-boosting style splitter using first/second moment stats and ``Criterion::gainAndHessian``.
 */

#include "Splitter.h"

class GainHessianSplitter : public Splitter<float, float> {
public:
  GainHessianSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                      arma::Mat<float> &y, double lambda)
      : Splitter(X, sampleWeights, 2), derivatives(y), lambda(lambda) {
    if (y.n_rows < 2)
      throw std::invalid_argument(
          "GainHessianSplitter requires y with at least 2 rows (gain, hessian)");
  }

  UnivariateSplitCandidate makeRoot() override;
  float predict(const UnivariateSplitCandidate &split) override;

  double score(const UnivariateSplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }
  double score(const std::vector<float> &stats, size_t l, size_t r) override;

protected:
  const arma::Mat<float> &derivatives;

  void moveSample(std::vector<float> &rightStats, std::vector<float> &leftStats,
                  size_t idx) override;

private:
  double lambda;
};
