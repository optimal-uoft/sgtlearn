#include "TreeBuilder.h"
template <typename T>
void UnivariateDiscretizer<T>::processLeaves(
    const std::map<std::tuple<size_t, size_t>, SplitCandidate> &leaves,
    Splitter<T> &splitter) {

  inSampleDiscretizations.clear();
  binPredictions.clear();
  thresholds.clear();
  for (auto &[_, leaf] : leaves) {
    inSampleDiscretizations.push_back(arma::conv_to<std::vector<size_t>>::from(
        training_.sortedOrder.subvec(leaf.start, leaf.end)));
    binPredictions.push_back(splitter.predict(leaf));
    thresholds.push_back(leaf.routingThreshold);
  }
  leavesProcessed = true;
}

template <typename T>
void UnivariateDiscretizer<T>::Train(Splitter<T> &splitter, size_t minLeafSize,
                                     double minGainSplit, size_t maxDepth,
                                     size_t maxLeafNodes) {



  TreeBuilder<SplitCandidate> treeBuilder(minLeafSize, minGainSplit, maxDepth,
                                          maxLeafNodes);
  std::map<std::tuple<size_t, size_t>, SplitCandidate> leaves;
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

  processLeaves(leaves, splitter);
  numLeaves = leaves.size();
}

template <typename T>
void UnivariateDiscretizer<T>::transform(const arma::fmat &X,
                                         arma::Row<size_t> &binLoc) {
  if (!leavesProcessed)
    throw std::runtime_error(
        "Cannot transform values without first training the discretizer");

  if (training_.feature >= X.n_rows)
    throw std::invalid_argument("Trained feature must be an index < X.n_rows");
  binLoc = arma::Row<size_t>(X.n_cols);
  const arma::frowvec featureRow = X.row(training_.feature);
  for (size_t col = 0; col < X.n_cols; ++col) {
    const auto it = std::ranges::lower_bound(thresholds, featureRow(col));
    binLoc(col) = std::distance(thresholds.begin(), it);
  }
}

template <typename T>
std::vector<std::vector<size_t>> &
UnivariateDiscretizer<T>::getInSampleDiscretizations() {
  if (!leavesProcessed)
    throw std::runtime_error("Cannot provide in sample routing without first "
                             "training the discretizer");
  return inSampleDiscretizations;
}

template <typename T>
std::vector<T> &UnivariateDiscretizer<T>::getBinPredictions() {
  if (!leavesProcessed)
    throw std::runtime_error("Cannot provide bin predictions without first "
                             "training the discretizer");
  return binPredictions;
}
