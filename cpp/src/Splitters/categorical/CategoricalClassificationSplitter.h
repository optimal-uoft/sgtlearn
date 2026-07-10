#pragma once

/**
 * @file CategoricalClassificationSplitter.h
 * @brief One-hot categorical classification splitter (Gini / entropy).
 */

#include "CategoricalSplitter.h"
#include "Criterion.h"
#include "Domain/LearningCriterion.h"

#include <armadillo>
#include <cstddef>
#include <stdexcept>
#include <vector>

class CategoricalClassificationSplitter
    : public CategoricalSplitter<double, size_t> {
public:
  CategoricalClassificationSplitter(
      const arma::fmat &X, const arma::Row<float> &sampleWeights,
      const arma::Row<size_t> &y, size_t numClasses,
      const std::vector<size_t> &featureIndices, LearningCriterion criterion)
      : CategoricalSplitter(X, sampleWeights, featureIndices), y_(y),
        numClasses_(numClasses), criterion_(criterion) {
    if (criterion_ != LearningCriterion::Gini &&
        criterion_ != LearningCriterion::Entropy)
      throw std::invalid_argument(
          "CategoricalClassificationSplitter requires Gini or Entropy");
  }

  size_t predict(const std::vector<size_t> &samples) override;

  double score(const std::vector<size_t> &samples) override;

  std::vector<double> statsForSamples(
      const std::vector<size_t> &samples) override;

private:
  const arma::Row<size_t> &y_;
  size_t numClasses_;
  LearningCriterion criterion_;

  std::vector<double> classCounts(const std::vector<size_t> &samples) const;
};
