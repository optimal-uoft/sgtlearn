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

class CategoricalRegressionSplitter : public CategoricalSplitter<double, float> {
public:
  CategoricalRegressionSplitter(
      const arma::fmat &X, const arma::Row<float> &sampleWeights,
      const arma::Row<float> &y, const std::vector<size_t> &featureIndices,
      LearningCriterion criterion)
      : CategoricalSplitter(X, sampleWeights, featureIndices), y_(y),
        criterion_(criterion) {
    if (criterion_ != LearningCriterion::SquaredError &&
        criterion_ != LearningCriterion::AbsoluteError)
      throw std::invalid_argument(
          "CategoricalRegressionSplitter requires SquaredError or AbsoluteError");
  }

  float predict(const std::vector<size_t> &samples) override;

  double score(const std::vector<size_t> &samples) override;

  std::vector<double> statsForSamples(
      const std::vector<size_t> &samples) override;

private:
  const arma::Row<float> &y_;
  LearningCriterion criterion_;

  std::vector<float> ysForSamples(const std::vector<size_t> &samples) const;
  std::vector<float> wsForSamples(const std::vector<size_t> &samples) const;
};
