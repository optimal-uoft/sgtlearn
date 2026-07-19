#pragma once

/**
 * @file CategoricalRegressionDiscretizer.h
 * @brief One-hot categorical regression discretizer (MSE / MAE).
 */

#include <cstddef>
#include "Discretizers/categorical/CategoricalDiscretizer.h"
#include "Discretizers/RegressionDiscretizer.h"
#include "Domain/LearningCriterion.h"
#include "Splitters/categorical/CategoricalRegressionSplitter.h"

#include <armadillo>
#include <stdexcept>

class CategoricalRegressionDiscretizer
    : public CategoricalDiscretizer<double, std::vector<float>>,
      public RegressionDiscretizer {
public:
  explicit CategoricalRegressionDiscretizer(
      LearningCriterion criterion = LearningCriterion::SquaredError)
      : criterion_(criterion) {
    if (criterion_ != LearningCriterion::SquaredError &&
        criterion_ != LearningCriterion::AbsoluteError)
      throw std::invalid_argument(
          "CategoricalRegressionDiscretizer requires SquaredError or "
          "AbsoluteError");
  }

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override {
    CategoricalDiscretizer<double, std::vector<float>>::transform(X, binLoc);
  }

  size_t routeToBin(const std::vector<float> &featureValues) const override {
    return CategoricalDiscretizer<double, std::vector<float>>::routeToBin(
        featureValues);
  }

  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Mat<float> &y, size_t minLeafSize, double minGainSplit,
             size_t maxDepth, size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights) override;

private:
  LearningCriterion criterion_;
};
