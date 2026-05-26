/**
 * @file GainHessianUnivariateDiscretizer.cpp
 * @brief Training implementation delegating to ``GainHessianSplitter``.
 */

#include "Discretizers/GainHessianUnivariateDiscretizer.h"

#include "Splitters/GainHessianSplitter.h"

namespace {

arma::frowvec sortedSampleWeights(const arma::uvec &sortedOrder,
                                  const arma::Row<float> &sampleWeights) {
  arma::frowvec w(sortedOrder.n_elem);
  if (sampleWeights.n_elem == 0) {
    w.ones();
    return w;
  }
  if (sampleWeights.n_elem != sortedOrder.n_elem)
    throw std::invalid_argument(
        "sample_weights length must equal number of training samples");
  for (arma::uword i = 0; i < sortedOrder.n_elem; ++i)
    w(i) = sampleWeights(sortedOrder(i));
  return w;
}

} // namespace

void GainHessianUnivariateDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::fmat &y, float lambda,
    size_t minLeafSize, double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  if (features.n_elem == 0)
    throw std::invalid_argument("features must contain at least one feature");
  if (y.n_rows != 2)
    throw std::invalid_argument(
        "y must be a 2xN matrix with row 0 = gain and row 1 = hessian");
  if (y.n_cols != X.n_cols)
    throw std::invalid_argument("y.n_cols must equal X.n_cols");
  if (features(0) >= X.n_rows)
    throw std::invalid_argument("features(0) must be < X.n_rows");

  feature = features(0);
  arma::uvec sortedOrder = arma::sort_index(X.row(feature));
  arma::fmat sortedY = y.cols(sortedOrder);
  arma::fmat XSorted = X.cols(sortedOrder);
  arma::frowvec sortedX = XSorted.row(feature);
  arma::frowvec sortedWeights =
      sortedSampleWeights(sortedOrder, sampleWeights);

  GainHessianSplitter splitter(sortedX, sortedWeights, sortedY,
                                 static_cast<double>(lambda));
  UnivariateDiscretizer<float, float>::buildTree(splitter, minLeafSize,
                                                 minGainSplit, maxDepth,
                                                 maxLeafNodes);
  UnivariateDiscretizer<float, float>::processLeaves(sortedOrder, splitter);
}
