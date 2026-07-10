/**
 * @file Discretizers/categorical/CategoricalRegressionDiscretizer.cpp
 */

#include "Discretizers/categorical/CategoricalRegressionDiscretizer.h"

#include <stdexcept>

namespace {

arma::Row<float> normalizedSampleWeights(const arma::fmat &X,
                                         const arma::Row<float> &sampleWeights) {
  if (sampleWeights.n_elem == 0) {
    arma::Row<float> w(X.n_cols);
    w.ones();
    return w;
  }
  if (sampleWeights.n_elem != X.n_cols)
    throw std::invalid_argument("sample_weights length must match X.n_cols");
  return sampleWeights;
}

void validateFeatureIndices(const arma::fmat &X, const arma::uvec &featureIndices) {
  for (size_t f : featureIndices) {
    if (f >= X.n_rows)
      throw std::invalid_argument("feature index out of range for X");
  }
}

} // namespace

void CategoricalRegressionDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<float> &y,
    size_t minLeafSize, double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  validateFeatureIndices(X, features);
  featureIndices_.assign(features.begin(), features.end());
  const arma::Row<float> w = normalizedSampleWeights(X, sampleWeights);

  CategoricalRegressionSplitter splitter(X, w, y, featureIndices_, criterion_);
  buildTree(X, splitter, minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
  processLeaves(splitter);
}
