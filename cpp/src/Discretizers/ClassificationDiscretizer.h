#pragma once

/**
 * @file ClassificationDiscretizer.h
 * @brief Classification task contract: ``Train`` on label rows plus post-training
 *        state from ``InnerDiscretizerBase<double>``.
 */

#include <cstddef>
#include "Discretizers/InnerDiscretizerBase.h"

#include <armadillo>

class ClassificationDiscretizer : public virtual InnerDiscretizerBase<double> {
public:
  ~ClassificationDiscretizer() override = default;

  virtual void Train(const arma::fmat &X, arma::uvec &features,
                     const arma::Row<size_t> &y, size_t numClasses,
                     size_t minLeafSize, double minGainSplit, size_t maxDepth,
                     size_t maxLeafNodes,
                     const arma::Row<float> &sampleWeights = arma::Row<float>()) = 0;
};
