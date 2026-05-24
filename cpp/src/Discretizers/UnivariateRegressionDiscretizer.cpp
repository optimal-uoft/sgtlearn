/**
 * @file UnivariateRegressionDiscretizer.cpp
 * @brief Explicit template instantiations for squared-error and MAE regression discretizers.
 */

#include "UnivariateRegressionDiscretizer.h"
#include "Splitters/AbsoluteErrorSplitter.h"

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

template <TRegressionSplitter Tsplitter>
void UnivariateRegressionDiscretizer<Tsplitter>::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<float> &y,
    size_t minLeafSize, double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  if (features(0) >= X.n_rows)
    throw std::invalid_argument("features(0) must be < X.n_rows");
  feature = features(0);
  arma::uvec sortedOrder = arma::sort_index(X.row(feature));
  arma::Mat<float> sortedY(1, sortedOrder.n_elem);
  for (arma::uword i = 0; i < sortedOrder.n_elem; ++i)
    sortedY(0, i) = y(sortedOrder(i));
  arma::fmat XSorted = X.cols(sortedOrder);
  arma::frowvec sortedX = XSorted.row(feature);
  arma::frowvec sortedWeights =
      sortedSampleWeights(sortedOrder, sampleWeights);

  Tsplitter splitter(sortedX, sortedWeights, sortedY);
  UnivariateDiscretizer<float, float>::buildTree(splitter, minLeafSize,
                                                 minGainSplit, maxDepth,
                                                 maxLeafNodes);
  UnivariateDiscretizer<float, float>::processLeaves(sortedOrder, splitter);
}

template class UnivariateRegressionDiscretizer<SquaredErrorSplitter>;
template class UnivariateRegressionDiscretizer<AbsoluteErrorSplitter>;
