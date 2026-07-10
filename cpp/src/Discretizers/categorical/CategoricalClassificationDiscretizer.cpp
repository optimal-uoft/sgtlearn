/**
 * @file Discretizers/categorical/CategoricalClassificationDiscretizer.cpp
 */

#include <cstddef>
#include "Discretizers/categorical/CategoricalClassificationDiscretizer.h"

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

void CategoricalClassificationDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<size_t> &y,
    size_t numClasses, size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes, const arma::Row<float> &sampleWeights) {
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  if (numClasses < 2)
    throw std::invalid_argument("numClasses must be >= 2");
  validateFeatureIndices(X, features);
  featureIndices_.assign(features.begin(), features.end());
  const arma::Row<float> w = normalizedSampleWeights(X, sampleWeights);

  CategoricalClassificationSplitter splitter(X, w, y, numClasses, featureIndices_,
                                             criterion_);
  buildTree(X, splitter, minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
  processLeaves(splitter);
}
