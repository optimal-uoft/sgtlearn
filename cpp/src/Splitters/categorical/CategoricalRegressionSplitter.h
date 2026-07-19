#pragma once

/**
 * @file CategoricalRegressionSplitter.h
 * @brief One-hot categorical regression splitter (MSE / MAE).
 */

#include "CategoricalSplitter.h"
#include "Criterion.h"
#include "Domain/LearningCriterion.h"

#include <armadillo>
#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * Multi-output one-hot categorical regression splitter. ``y`` has shape
 * ``(nOutputs, nSamples)``; MSE stats are ``[Σw·y0, Σw·y0², ...]`` (length
 * ``2 * nOutputs``) and both MSE and MAE scores sum per-output loss.
 * ``predict`` returns the per-output mean (MSE) or median (MAE).
 */
class CategoricalRegressionSplitter
    : public CategoricalSplitter<double, std::vector<float>> {
public:
  CategoricalRegressionSplitter(
      const arma::fmat &X, const arma::Row<float> &sampleWeights,
      const arma::Mat<float> &y, const std::vector<size_t> &featureIndices,
      LearningCriterion criterion)
      : CategoricalSplitter(X, sampleWeights, featureIndices), y_(y),
        nOutputs_(static_cast<size_t>(y.n_rows)), criterion_(criterion) {
    if (criterion_ != LearningCriterion::SquaredError &&
        criterion_ != LearningCriterion::AbsoluteError)
      throw std::invalid_argument(
          "CategoricalRegressionSplitter requires SquaredError or AbsoluteError");
  }

  std::vector<float> predict(const std::vector<size_t> &samples) override;

  double score(const std::vector<size_t> &samples) override;

  std::vector<double> statsForSamples(
      const std::vector<size_t> &samples) override;

private:
  const arma::Mat<float> &y_;
  size_t nOutputs_;
  LearningCriterion criterion_;

  std::vector<float> ysForSamples(size_t output,
                                  const std::vector<size_t> &samples) const;
  std::vector<float> wsForSamples(const std::vector<size_t> &samples) const;
};
