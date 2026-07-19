#pragma once

/**
 * @file UnivariateRegressionDiscretizer.h
 * @brief Numeric univariate regression discretizer parameterized by splitter.
 */

#include <cstddef>
#include "Discretizers/RegressionDiscretizer.h"
#include "Discretizers/univariate/UnivariateDiscretizer.h"
#include "Splitters/univariate/AbsoluteErrorSplitter.h"
#include "Splitters/univariate/SquaredErrorSplitter.h"

#include <armadillo>

template <typename T>
concept TRegressionSplitter =
    std::constructible_from<T, arma::frowvec &, arma::frowvec &,
                            arma::Mat<float> &>;

template <TRegressionSplitter TSplitter>
class UnivariateRegressionDiscretizer
    : public UnivariateDiscretizer<double, std::vector<float>>,
      public RegressionDiscretizer {
public:
  ~UnivariateRegressionDiscretizer() override = default;

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override {
    UnivariateDiscretizer<double, std::vector<float>>::transform(X, binLoc);
  }

  size_t routeToBin(const std::vector<float> &featureValues) const override {
    return UnivariateDiscretizer<double, std::vector<float>>::routeToBin(
        featureValues);
  }

  void Train(const arma::fmat &X, arma::uvec &features, const arma::Mat<float> &y,
             size_t minLeafSize, double minGainSplit, size_t maxDepth,
             size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights) override;
};
