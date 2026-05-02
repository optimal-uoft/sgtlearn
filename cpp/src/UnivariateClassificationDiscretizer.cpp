#include "UnivariateClassificationDiscretizer.h"
#include "frontiers.h"
#include <algorithm>
#include <armadillo>
#include <map>
#include <stdexcept>

#include "GiniSplitter.h"
#include "SplitCandidate.h"
#include "TreeBuilder.h"

UnivariateClassificationDiscretizer::~UnivariateClassificationDiscretizer() =
    default;

void UnivariateClassificationDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<size_t> &y,
    size_t numClasses, size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes) {

  // region param validation
  if (X.n_rows == 0)
    throw std::invalid_argument("X must have at least one feature row");
  if (features.n_elem != 1)
    throw std::invalid_argument("features must be size 1");
  if (features(0) >= X.n_rows)
    throw std::invalid_argument("features(0) must be < X.n_rows");
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  // endregion

  arma::uvec sortedOrder = arma::sort_index(X.row(features(0)));
  arma::Row sortedY = y.cols(sortedOrder);
  arma::fmat XSorted = X.cols(sortedOrder);
  arma::frowvec sortedX = XSorted.row(features(0));

  TreeBuilder<SplitCandidate> treeBuilder(minLeafSize, minGainSplit, maxDepth,
                                          maxLeafNodes);
  std::map<std::tuple<size_t, size_t>, SplitCandidate> leaves;

  GiniSplitter splitter(sortedX, sortedY, numClasses);
  SplitCandidate rootSplit = splitter.makeRoot();
  leaves[std::make_tuple(rootSplit.start, rootSplit.end)] = rootSplit;

  treeBuilder.buildTree(
      rootSplit,
      [&splitter](SplitCandidate &split, size_t minLeafSize) {
        return splitter.findBestSplit(split, minLeafSize);
      },
      [&splitter](SplitCandidate &split) {
        return splitter.makeChildren(split);
      },
      [&leaves](SplitCandidate &split, std::vector<SplitCandidate> &children) {
        leaves.erase(std::make_tuple(split.start, split.end));
        for (auto child : children)
          leaves[std::make_tuple(child.start, child.end)] = child;
      });

  inSampleDiscretizations.clear();
  binPredictions.clear();
  std::vector<double> thresholds;
  for (auto &[_, leaf] : leaves) {
    inSampleDiscretizations.push_back(arma::conv_to<std::vector<size_t>>::from(
        sortedOrder.subvec(leaf.start, leaf.end)));
    binPredictions.push_back(splitter.predict(leaf));
    thresholds.push_back(leaf.routingThreshold);
  }

  binMapFunction = [features, thresholds](const arma::fmat &XNew,
                                          arma::Row<size_t> &binLoc) {
    if (features(0) >= XNew.n_rows)
      throw std::invalid_argument("features(0) must be < XNew.n_rows");
    binLoc = arma::Row<size_t>(XNew.n_cols);
    const arma::frowvec featureRow = XNew.row(features(0));
    for (size_t col = 0; col < XNew.n_cols; ++col) {
      const auto it = std::ranges::lower_bound(thresholds, featureRow(col));
      binLoc(col) = std::distance(thresholds.begin(), it);
    }
  };

  numLeaves = leaves.size();
}

void UnivariateClassificationDiscretizer::transform(const arma::fmat &X,
                                                    arma::Row<size_t> &binLoc) {
  if (binMapFunction == nullptr)
    throw std::runtime_error(
        "Cannot transform values without first training the discretizer");
  binMapFunction(X, binLoc);
}

const std::vector<std::vector<size_t>> &
UnivariateClassificationDiscretizer::getInSampleDiscretizations() {
  if (binMapFunction == nullptr)
    throw std::runtime_error(
        "Cannot transform values without first training the discretizer");
  return inSampleDiscretizations;
}

const std::vector<size_t> &
UnivariateClassificationDiscretizer::getBinPredictions() {
  if (binMapFunction == nullptr)
    throw std::runtime_error(
        "Cannot transform values without first training the discretizer");
  return binPredictions;
}
