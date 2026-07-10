#pragma once

/**
 * @file GainHessianUnivariateDiscretizer.h
 * @brief Univariate discretizer for gradient boosting targets (per-sample gradient and hessian rows in ``y``).
 */

#include "Discretizers/univariate/UnivariateDiscretizer.h"

#include <armadillo>

class GainHessianUnivariateDiscretizer
    : public UnivariateDiscretizer<float, float> {
public:
  ~GainHessianUnivariateDiscretizer() = default;

  /** ``y`` is a 2-row matrix: gradients and hessians per sample; ``lambda`` is L2 leaf regularization. */
  void Train(const arma::fmat &X, arma::uvec &features, const arma::fmat &y,
             float lambda, size_t minLeafSize = 1,
             double minGainSplit = 1e-7, size_t maxDepth = 0,
             size_t maxLeafNodes = 0,
             const arma::Row<float> &sampleWeights = arma::Row<float>());
};
