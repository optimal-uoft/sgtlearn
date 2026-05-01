#include "UnivariateClassificationDiscretizer.h"
#include "frontiers.h"
#include <algorithm>
#include <armadillo>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

#include "GiniSplitter.h"
#include "SplitCandidate.h"

constexpr double eps = std::numeric_limits<double>::epsilon();

UnivariateClassificationDiscretizer::~UnivariateClassificationDiscretizer() =
    default;

void UnivariateClassificationDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<size_t> &y,
    size_t numClasses, size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes) {
  if (X.n_rows == 0)
    throw std::invalid_argument("X must have at least one feature row");
  if (X.n_cols == 0)
    throw std::invalid_argument("X must have at least one sample column");
  if (features.n_elem != 1)
    throw std::invalid_argument("features must be size 1");
  if (features(0) >= X.n_rows)
    throw std::invalid_argument("features(0) must be < X.n_rows");
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  if (numClasses == 0)
    throw std::invalid_argument("numClasses must be > 0");
  if (minLeafSize == 0)
    throw std::invalid_argument("minLeafSize must be > 0");

  arma::uvec sortedOrder = arma::sort_index(X.row(features(0)));
  arma::Row sortedY = y.cols(sortedOrder);
  arma::fmat XSorted = X.cols(sortedOrder);
  arma::frowvec sortedX = XSorted.row(features(0));
  

  GiniSplitter splitter(numClasses);

  std::unique_ptr<frontiers::IFrontier<SplitCandidate>> frontier;
  if (maxLeafNodes == 0)
    frontier = std::make_unique<frontiers::Stack<SplitCandidate>>();
  else
    frontier = std::make_unique<frontiers::Heap<SplitCandidate>>();
  std::map<std::tuple<size_t, size_t>, SplitCandidate> leaves;

  SplitCandidate rootSplit = splitter.makeRoot(sortedY);

  if (rootSplit.score > eps &&
      splitter.findBestSplit(rootSplit, sortedX, sortedY, minLeafSize))
    frontier->push(rootSplit);
  leaves[std::make_tuple(rootSplit.start, rootSplit.end)] = rootSplit;

  while (frontier->size() > 0 &&
         (maxLeafNodes == 0 || leaves.size() < maxLeafNodes)) {
    auto split = frontier->peek();
    frontier->pop();

    if (split.score <= eps || split.informationGain + eps < minGainSplit ||
        (maxDepth != 0 && split.height >= maxDepth)) {
      continue;
    }
    auto children = splitter.makeChildren(split);
    auto left = children[0];
    auto right = children[1];

    if (right.score > eps &&
        splitter.findBestSplit(right, sortedX, sortedY, minLeafSize))
      frontier->push(right);
    if (left.score > eps &&
        splitter.findBestSplit(left, sortedX, sortedY, minLeafSize))
      frontier->push(left);

    leaves.erase(std::make_tuple(split.start, split.end));
    leaves[std::make_tuple(left.start, left.end)] = left;
    leaves[std::make_tuple(right.start, right.end)] = right;
  }

  inSampleDiscretizations.clear();
  binPredictions.clear();
  std::vector<double> thresholds;


  for (auto &[_, leaf]: leaves) {
    inSampleDiscretizations.push_back(
        arma::conv_to<std::vector<size_t> >::from(sortedOrder.subvec(leaf.start, leaf.end)));
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
