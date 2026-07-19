#pragma once

/**
 * @file ClassificationDiscretizer.h
 * @brief Classification task contract: ``Train`` on label rows plus post-training
 *        state from ``InnerDiscretizerBase<double>``.
 */

#include <cstddef>
#include <vector>
#include "Discretizers/InnerDiscretizerBase.h"

#include <armadillo>

class ClassificationDiscretizer : public virtual InnerDiscretizerBase<double> {
public:
  ~ClassificationDiscretizer() override = default;

  /**
   * @param y                 label matrix, shape ``(nOutputs, nSamples)``;
   *                          single-output is ``nOutputs == 1``. A
   *                          ``Row<size_t>`` argument converts implicitly.
   * @param nClassesPerOutput number of classes for each output row.
   */
  virtual void Train(const arma::fmat &X, arma::uvec &features,
                     const arma::Mat<size_t> &y,
                     const std::vector<size_t> &nClassesPerOutput,
                     size_t minLeafSize, double minGainSplit, size_t maxDepth,
                     size_t maxLeafNodes,
                     const arma::Row<float> &sampleWeights = arma::Row<float>()) = 0;

  /**
   * Convenience overload for a single output (or the same class count across a
   * ``y`` with ``nOutputs`` rows): expands ``numClasses`` to one entry per
   * output row of ``y`` and forwards to the multi-output ``Train``.
   */
  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Mat<size_t> &y, size_t numClasses, size_t minLeafSize,
             double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights = arma::Row<float>()) {
    const size_t nOutputs = y.n_rows == 0 ? 1 : static_cast<size_t>(y.n_rows);
    Train(X, features, y, std::vector<size_t>(nOutputs, numClasses), minLeafSize,
          minGainSplit, maxDepth, maxLeafNodes, sampleWeights);
  }
};
