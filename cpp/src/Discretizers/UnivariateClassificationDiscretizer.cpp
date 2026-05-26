/**
 * @file UnivariateClassificationDiscretizer.cpp
 * @brief Explicit template instantiations for Gini and entropy classification discretizers.
 */

#include "Discretizers/UnivariateClassificationDiscretizer.h"
#include "Splitters/EntropySplitter.h"

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

template <TClassificationSplitter Tsplitter>
void UnivariateClassificationDiscretizer<Tsplitter>::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<size_t> &y,
    size_t numClasses, size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes, const arma::Row<float> &sampleWeights) {
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  if (features(0) >= X.n_rows)
    throw std::invalid_argument("features(0) must be < X.n_rows");
  feature = features(0);
  arma::uvec sortedOrder = arma::sort_index(X.row(feature));
  arma::Mat<size_t> sortedY(1, sortedOrder.n_elem);
  for (arma::uword i = 0; i < sortedOrder.n_elem; ++i)
    sortedY(0, i) = y(sortedOrder(i));
  arma::fmat XSorted = X.cols(sortedOrder);
  arma::frowvec sortedX = XSorted.row(feature);
  arma::frowvec sortedWeights =
      sortedSampleWeights(sortedOrder, sampleWeights);

  Tsplitter splitter(sortedX, sortedWeights, sortedY, numClasses);
  UnivariateDiscretizer<double, size_t>::buildTree(splitter, minLeafSize,
                                                   minGainSplit, maxDepth,
                                                   maxLeafNodes);

  UnivariateDiscretizer<double, size_t>::processLeaves(sortedOrder, splitter);
}

template class UnivariateClassificationDiscretizer<GiniSplitter>;
template class UnivariateClassificationDiscretizer<EntropySplitter>;
