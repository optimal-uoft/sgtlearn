#pragma once

/**
 * @file CategoricalClassificationDiscretizer.h
 * @brief One-hot categorical classification discretizer (Gini / entropy).
 */

#include <cstddef>
#include "Discretizers/categorical/CategoricalDiscretizer.h"
#include "Discretizers/ClassificationDiscretizer.h"
#include "Domain/LearningCriterion.h"
#include "Splitters/categorical/CategoricalClassificationSplitter.h"

#include <armadillo>
#include <stdexcept>

class CategoricalClassificationDiscretizer
    : public CategoricalDiscretizer<double, size_t>,
      public ClassificationDiscretizer {
public:
  explicit CategoricalClassificationDiscretizer(
      LearningCriterion criterion = LearningCriterion::Gini)
      : criterion_(criterion) {
    if (criterion_ != LearningCriterion::Gini &&
        criterion_ != LearningCriterion::Entropy)
      throw std::invalid_argument(
          "CategoricalClassificationDiscretizer requires Gini or Entropy");
  }

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override {
    CategoricalDiscretizer<double, size_t>::transform(X, binLoc);
  }

  size_t routeToBin(const std::vector<float> &featureValues) const override {
    return CategoricalDiscretizer<double, size_t>::routeToBin(featureValues);
  }

  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Row<size_t> &y, size_t numClasses, size_t minLeafSize,
             double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights = arma::Row<float>()) override;

private:
  LearningCriterion criterion_;
};
