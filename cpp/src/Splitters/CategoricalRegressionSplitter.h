#pragma once

/**
 * @file CategoricalRegressionSplitter.h
 * @brief One-hot categorical regression splitter (MSE / MAE).
 */

#include "CategoricalSplitter.h"

#include "Criterion.h"

#include <armadillo>
#include <cstddef>
#include <vector>

enum class CategoricalRegressionCriterion { SquaredError, AbsoluteError };

class CategoricalRegressionSplitter : public CategoricalSplitter<double, float> {
public:
  CategoricalRegressionSplitter(
      const arma::fmat &X, const arma::Row<float> &sampleWeights,
      const arma::Row<float> &y, const std::vector<size_t> &featureIndices,
      CategoricalRegressionCriterion criterion)
      : CategoricalSplitter(X, sampleWeights, featureIndices), y_(y),
        criterion_(criterion) {}

  float predict(const std::vector<size_t> &samples) override;

  double score(const std::vector<size_t> &samples) override;

  std::vector<double> statsForSamples(
      const std::vector<size_t> &samples) override;

private:
  const arma::Row<float> &y_;
  CategoricalRegressionCriterion criterion_;

  std::vector<float> ysForSamples(const std::vector<size_t> &samples) const;
  std::vector<float> wsForSamples(const std::vector<size_t> &samples) const;
};
