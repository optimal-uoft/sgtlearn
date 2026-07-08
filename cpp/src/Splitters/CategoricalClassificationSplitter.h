#pragma once

/**
 * @file CategoricalClassificationSplitter.h
 * @brief One-hot categorical classification splitter (Gini / entropy).
 */

#include "CategoricalSplitter.h"

#include "Criterion.h"

#include <armadillo>
#include <cstddef>
#include <vector>

enum class CategoricalClassificationCriterion { Gini, Entropy };

class CategoricalClassificationSplitter
    : public CategoricalSplitter<double, size_t> {
public:
  CategoricalClassificationSplitter(
      const arma::fmat &X, const arma::Row<float> &sampleWeights,
      const arma::Row<size_t> &y, size_t numClasses,
      const std::vector<size_t> &featureIndices,
      CategoricalClassificationCriterion criterion)
      : CategoricalSplitter(X, sampleWeights, featureIndices), y_(y),
        numClasses_(numClasses), criterion_(criterion) {}

  size_t predict(const std::vector<size_t> &samples) override;

  double score(const std::vector<size_t> &samples) override;

  std::vector<double> statsForSamples(
      const std::vector<size_t> &samples) override;

private:
  const arma::Row<size_t> &y_;
  size_t numClasses_;
  CategoricalClassificationCriterion criterion_;

  std::vector<double> classCounts(const std::vector<size_t> &samples) const;
};
