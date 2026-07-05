/**
 * @file UnivariateRegressionDiscretizer.cpp
 * @brief Explicit template instantiations for squared-error and MAE regression discretizers.
 */

#include "algorithms/missing_values.h"
#include "Discretizers/UnivariateRegressionDiscretizer.h"
#include "Splitters/AbsoluteErrorSplitter.h"

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

template <TRegressionSplitter TSplitter>
void UnivariateRegressionDiscretizer<TSplitter>::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<float> &y,
    size_t minLeafSize, double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  if (features(0) >= X.n_rows)
    throw std::invalid_argument("features(0) must be < X.n_rows");
  this->feature = features(0);
  const auto sort =
      missing_values::sort_index_finite_first(X.row(this->feature));
  const arma::uword n_finite =
      static_cast<arma::uword>(sort.first_non_finite_index);

  // NaN bucket: aggregate the non-finite tail as ``[sum w*y, sum w*y^2]``. The
  // inner tree below is fit on finite values only, so the splitter operates on
  // N_numeric = N - N_nan samples and min_samples_leaf applies to numerics.
  this->nanSeen_ = (n_finite < X.n_cols);
  this->nanNumSamples_ = 0;
  this->nanNodeWeight_ = 0.0;
  this->nanInSampleIndices_.clear();
  double nanSumWY = 0.0;
  double nanSumWY2 = 0.0;
  for (arma::uword i = n_finite; i < X.n_cols; ++i) {
    const arma::uword idx = sort.order(i);
    const double w =
        sampleWeights.n_elem == 0 ? 1.0 : static_cast<double>(sampleWeights(idx));
    const double v = static_cast<double>(y(idx));
    nanSumWY += w * v;
    nanSumWY2 += w * v * v;
    this->nanNodeWeight_ += w;
    ++this->nanNumSamples_;
    this->nanInSampleIndices_.push_back(static_cast<size_t>(idx));
  }
  this->nanStats_ = {nanSumWY, nanSumWY2};

  if (n_finite == 0)
    return;
  const arma::uvec finiteOrder = sort.order.subvec(0, n_finite - 1);
  arma::Mat<float> sortedY(1, n_finite);
  for (arma::uword i = 0; i < n_finite; ++i)
    sortedY(0, i) = y(finiteOrder(i));
  arma::fmat XSorted = X.cols(finiteOrder);
  arma::frowvec sortedX = XSorted.row(this->feature);
  arma::frowvec sortedWeights =
      sortedSampleWeights(finiteOrder, sampleWeights);

  TSplitter splitter(sortedX, sortedWeights, sortedY);
  this->buildTree(splitter, minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
  this->processLeaves(finiteOrder, splitter);
}

template class UnivariateRegressionDiscretizer<SquaredErrorSplitter>;
template class UnivariateRegressionDiscretizer<AbsoluteErrorSplitter>;
