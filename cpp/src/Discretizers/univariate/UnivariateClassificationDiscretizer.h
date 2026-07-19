#pragma once

/**
 * @file UnivariateClassificationDiscretizer.h
 * @brief Numeric univariate classification discretizer parameterized by splitter.
 */

#include <cstddef>
#include "Discretizers/ClassificationDiscretizer.h"
#include "Discretizers/univariate/UnivariateDiscretizer.h"
#include "Splitters/univariate/ClassificationSplitter.h"
#include "Splitters/univariate/EntropySplitter.h"
#include "Splitters/univariate/GiniSplitter.h"

#include <armadillo>

template <typename T>
concept TClassificationSplitter =
    std::derived_from<T, ClassificationSplitter> &&
    std::constructible_from<T, arma::frowvec &, arma::frowvec &,
                            arma::Mat<size_t> &, const std::vector<size_t> &>;

template <TClassificationSplitter Tsplitter = GiniSplitter>
class UnivariateClassificationDiscretizer
    : public UnivariateDiscretizer<double, std::vector<size_t>>,
      public ClassificationDiscretizer {
public:
  ~UnivariateClassificationDiscretizer() override = default;

  using ClassificationDiscretizer::Train;

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override {
    UnivariateDiscretizer<double, std::vector<size_t>>::transform(X, binLoc);
  }

  size_t routeToBin(const std::vector<float> &featureValues) const override {
    return UnivariateDiscretizer<double, std::vector<size_t>>::routeToBin(
        featureValues);
  }

  void Train(const arma::fmat &X, arma::uvec &features, const arma::Mat<size_t> &y,
             const std::vector<size_t> &nClassesPerOutput, size_t minLeafSize = 1,
             double minGainSplit = 1e-7, size_t maxDepth = 0,
             size_t maxLeafNodes = 0,
             const arma::Row<float> &sampleWeights = arma::Row<float>()) override;
};
