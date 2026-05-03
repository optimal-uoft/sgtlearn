#include "Splitters/EntropySplitter.h"
#include "UnivariateClassificationDiscretizer.h"

template <TClassificationSplitter Tsplitter>
void UnivariateClassificationDiscretizer<Tsplitter>::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<size_t> &y,
    size_t numClasses, size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes) {
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  if (features(0) >= X.n_rows)
    throw std::invalid_argument("features(0) must be < X.n_rows");

  arma::uvec sortedOrder = arma::sort_index(X.row(features(0)));
  arma::Row<size_t> sortedY = y.cols(sortedOrder);
  arma::fmat XSorted = X.cols(sortedOrder);
  arma::frowvec sortedX = XSorted.row(features(0));
  setTrainingContext(
      {std::move(sortedX), std::move(sortedY), std::move(sortedOrder),
       features(0)});
  Tsplitter splitter(training_.sortedX, training_.sortedY, numClasses);
  UnivariateDiscretizer<size_t>::Train(splitter, minLeafSize, minGainSplit,
                                       maxDepth, maxLeafNodes);
}

template class UnivariateClassificationDiscretizer<GiniSplitter>;
template class UnivariateClassificationDiscretizer<EntropySplitter>;
