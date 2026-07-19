#pragma once

/**
 * @file RegressionDiscretizer.h
 * @brief Regression task contract: ``Train`` on target rows plus post-training
 *        state from ``InnerDiscretizerBase<double>``.
 */

#include <cstddef>
#include "Discretizers/InnerDiscretizerBase.h"

#include <armadillo>

class RegressionDiscretizer : public virtual InnerDiscretizerBase<double> {
public:
  ~RegressionDiscretizer() override = default;

  /**
   * @param y target matrix, shape ``(nOutputs, nSamples)``; single-output is
   *          ``nOutputs == 1``. A ``Row<float>`` argument converts implicitly.
   */
  virtual void Train(const arma::fmat &X, arma::uvec &features,
                     const arma::Mat<float> &y, size_t minLeafSize,
                     double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
                     const arma::Row<float> &sampleWeights) = 0;
};
