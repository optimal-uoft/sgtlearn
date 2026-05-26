#pragma once

/**
 * @file UnivariateRegressionDiscretizer.h
 * @brief One-feature regression discretizer (MSE / MAE / related splitters) producing piecewise-constant bin predictions.
 */

#include "Discretizers/UnivariateDiscretizer.h"
#include "Splitters/AbsoluteErrorSplitter.h"
#include "Splitters/SquaredErrorSplitter.h"

#include <armadillo>

template <typename T>
concept TRegressionSplitter =
    std::constructible_from<T, arma::frowvec &, arma::frowvec &,
                            arma::Mat<float> &>;

template <TRegressionSplitter TSplitter>
class UnivariateRegressionDiscretizer
    : public UnivariateDiscretizer<double, float> {
public:
  ~UnivariateRegressionDiscretizer() = default;

  /** Fit inner regression tree on selected feature rows of ``X`` with targets ``y``. */
  void Train(const arma::fmat &X, arma::uvec &features, const arma::Row<float> &y,
             size_t minLeafSize, double minGainSplit, size_t maxDepth,
             size_t maxLeafNodes, const arma::Row<float> &sampleWeights);
};
