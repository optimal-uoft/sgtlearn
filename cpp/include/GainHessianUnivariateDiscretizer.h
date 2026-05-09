#pragma once

#include "Discretizers/UnivariateDiscretizer.h"

#include <armadillo>

class GainHessianUnivariateDiscretizer : public UnivariateDiscretizer<float> {
public:
  ~GainHessianUnivariateDiscretizer() = default;

  void Train(const arma::fmat &X, arma::uvec &features, const arma::fmat &y,
             float lambda, size_t minLeafSize = 1,
             double minGainSplit = 1e-7, size_t maxDepth = 0,
             size_t maxLeafNodes = 0);
};
