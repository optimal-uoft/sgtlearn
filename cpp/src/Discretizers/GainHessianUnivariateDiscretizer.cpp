/**
 * @file GainHessianUnivariateDiscretizer.cpp
 * @brief Training implementation delegating to ``GainHessianSplitter``.
 */

#include "algorithms/missing_values.h"

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
  for (arma::uword i = 0; i < sortedOrder.n_elem; ++i) {
    const arma::uword idx = sortedOrder(i);
    if (idx >= sampleWeights.n_elem)
      throw std::invalid_argument(
          "sample_weights index out of range for training sample order");
    w(i) = sampleWeights(idx);
  }
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
  const auto sort = missing_values::sort_index_finite_first(X.row(feature));
  const arma::uword n_finite =
      static_cast<arma::uword>(sort.first_non_finite_index);
  if (n_finite == 0)
    return;
  const arma::uvec finiteOrder = sort.order.subvec(0, n_finite - 1);
  arma::fmat sortedY = y.cols(finiteOrder);
  arma::fmat XSorted = X.cols(finiteOrder);
  arma::frowvec sortedX = XSorted.row(feature);
  arma::frowvec sortedWeights =
      sortedSampleWeights(finiteOrder, sampleWeights);

  GainHessianSplitter splitter(sortedX, sortedWeights, sortedY,
                               static_cast<double>(lambda));
  this->buildTree(splitter, minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
  this->processLeaves(finiteOrder, splitter);
}
