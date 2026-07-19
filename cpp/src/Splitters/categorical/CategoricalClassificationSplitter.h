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

/**
 * Multi-output one-hot categorical classification splitter. ``y`` has shape
 * ``(nOutputs, nSamples)``; stats are concatenated per-output weighted class
 * histograms and the score is the SUM of per-output Gini/entropy. ``predict``
 * returns the per-output argmax.
 */
class CategoricalClassificationSplitter
    : public CategoricalSplitter<double, std::vector<size_t>> {
public:
  CategoricalClassificationSplitter(
      const arma::fmat &X, const arma::Row<float> &sampleWeights,
      const arma::Mat<size_t> &y,
      const std::vector<size_t> &nClassesPerOutput,
      const std::vector<size_t> &featureIndices, LearningCriterion criterion)
      : CategoricalSplitter(X, sampleWeights, featureIndices), y_(y),
        classesPerOutput_(nClassesPerOutput),
        nOutputs_(static_cast<size_t>(y.n_rows)), criterion_(criterion) {
    if (criterion_ != LearningCriterion::Gini &&
        criterion_ != LearningCriterion::Entropy)
      throw std::invalid_argument(
          "CategoricalClassificationSplitter requires Gini or Entropy");
    if (classesPerOutput_.size() != nOutputs_)
      throw std::invalid_argument(
          "nClassesPerOutput length must equal y.n_rows");
    classOffsets_.assign(nOutputs_, 0);
    totalClasses_ = 0;
    for (size_t o = 0; o < nOutputs_; ++o) {
      classOffsets_[o] = totalClasses_;
      totalClasses_ += classesPerOutput_[o];
    }
  }

  std::vector<size_t> predict(const std::vector<size_t> &samples) override;

  double score(const std::vector<size_t> &samples) override;

  std::vector<double> statsForSamples(
      const std::vector<size_t> &samples) override;

private:
  const arma::Mat<size_t> &y_;
  std::vector<size_t> classesPerOutput_;
  std::vector<size_t> classOffsets_;
  size_t nOutputs_;
  size_t totalClasses_ = 0;
  LearningCriterion criterion_;

  std::vector<double> classCounts(const std::vector<size_t> &samples) const;
};
