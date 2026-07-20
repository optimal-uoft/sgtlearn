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
 * ``(nOutputs, nSamples)``; stats are nested per-output weighted class
 * histograms ``[output][class]`` and the score is the SUM of per-output
 * Gini/entropy. ``predict`` returns the per-output argmax.
 */
class CategoricalClassificationSplitter
    : public CategoricalSplitter<std::vector<double>, std::vector<size_t>> {
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
  }

  std::vector<size_t> predict(const std::vector<size_t> &samples) override;

  double score(const std::vector<size_t> &samples) override;

  std::vector<std::vector<double>> statsForSamples(
      const std::vector<size_t> &samples) override;

private:
  const arma::Mat<size_t> &y_;
  std::vector<size_t> classesPerOutput_;
  size_t nOutputs_;
  LearningCriterion criterion_;

  std::vector<std::vector<double>>
  classCounts(const std::vector<size_t> &samples) const;
};
