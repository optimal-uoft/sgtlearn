#pragma once

/**
 * @file RegressionDiscretizer.h
 * @brief Regression task contract: ``Train`` on target rows plus post-training
 *        state from ``InnerDiscretizerBase<double>``.
 */

#include "Discretizers/InnerDiscretizerBase.h"

#include <armadillo>

class RegressionDiscretizer : public virtual InnerDiscretizerBase<double> {
public:
  ~RegressionDiscretizer() override = default;

  virtual void Train(const arma::fmat &X, arma::uvec &features,
                     const arma::Row<float> &y, size_t minLeafSize,
                     double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
                     const arma::Row<float> &sampleWeights) = 0;
};
