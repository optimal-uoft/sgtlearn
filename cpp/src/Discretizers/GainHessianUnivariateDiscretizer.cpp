#include "GainHessianUnivariateDiscretizer.h"

#include "Splitters/GainHessianSplitter.h"

void GainHessianUnivariateDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::fmat &y,
    float lambda, size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes) {
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

  GainHessianSplitter splitter(sortedX, sortedY, static_cast<double>(lambda));
  UnivariateDiscretizer<float>::buildTree(splitter, minLeafSize, minGainSplit,
                                          maxDepth, maxLeafNodes);
  UnivariateDiscretizer::processLeaves(sortedOrder, splitter);
}
