/**
 * @file UnivariateDiscretizer.tpp
 * @brief Template implementation for ``UnivariateDiscretizer`` training and ``transform``.
 */

#include "../algorithms/TreeBuilder.h"

template <typename StatsT, typename PredictT>
void UnivariateDiscretizer<StatsT, PredictT>::processLeaves(
    arma::uvec sortedOrder, Splitter<StatsT, PredictT> &splitter) {
  if (step != Step::FitTree)
    throw std::runtime_error("tree must be fit before leaves are processed");
  inSampleDiscretizations.clear();
  binPredictions.clear();
  thresholds.clear();
  leafStats.clear();
  leafNumSamples.clear();
  leafNodeWeights.clear();
  for (auto &[_, leaf] : leaves) {
    inSampleDiscretizations.push_back(arma::conv_to<std::vector<size_t>>::from(
        sortedOrder.subvec(leaf.start, leaf.end)));
    binPredictions.push_back(splitter.predict(leaf));
    leafStats.push_back(splitter.getStats(leaf));
    leafNumSamples.push_back(leaf.numSamples);
    leafNodeWeights.push_back(leaf.nodeWeight);
    thresholds.push_back(leaf.routingThreshold);
  }
  step = Step::LeavesProcessed;
}

template <typename StatsT, typename PredictT>
void UnivariateDiscretizer<StatsT, PredictT>::buildTree(
    Splitter<StatsT, PredictT> &splitter, size_t minLeafSize,
    double minGainSplit, size_t maxDepth, size_t maxLeafNodes) {
  leaves.clear();
  TreeBuilder<SplitCandidate> treeBuilder(minLeafSize, minGainSplit, maxDepth,
                                          maxLeafNodes);
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
      [this](SplitCandidate &split, std::vector<SplitCandidate> &children) {
        leaves.erase(std::make_tuple(split.start, split.end));
        for (auto child : children)
          leaves[std::make_tuple(child.start, child.end)] = child;
      });
  step = Step::FitTree;
  numLeaves = leaves.size();
}

template <typename StatsT, typename PredictT>
void UnivariateDiscretizer<StatsT, PredictT>::transform(const arma::fmat &X,
                                                      arma::Row<size_t> &binLoc) {
  if (step != Step::LeavesProcessed)
    throw std::runtime_error(
        "Cannot transform values without first training the discretizer");

  if (feature >= X.n_rows)
    throw std::invalid_argument("Trained feature must be an index < X.n_rows");
  binLoc = arma::Row<size_t>(X.n_cols);
  const arma::frowvec featureRow = X.row(feature);
  for (size_t col = 0; col < X.n_cols; ++col) {
    const auto it = std::ranges::lower_bound(thresholds, featureRow(col));
    binLoc(col) = std::distance(thresholds.begin(), it);
  }
}

template <typename StatsT, typename PredictT>
std::vector<std::vector<size_t>> &
UnivariateDiscretizer<StatsT, PredictT>::getInSampleDiscretizations() {
  if (step != Step::LeavesProcessed)
    throw std::runtime_error("Cannot provide in sample routing without first "
                             "training the discretizer");
  return inSampleDiscretizations;
}

template <typename StatsT, typename PredictT>
std::vector<PredictT> &UnivariateDiscretizer<StatsT, PredictT>::getBinPredictions() {
  if (step != Step::LeavesProcessed)
    throw std::runtime_error("Cannot provide bin predictions without first "
                             "training the discretizer");
  return binPredictions;
}
template <typename StatsT, typename PredictT>
std::vector<std::vector<StatsT>> &
UnivariateDiscretizer<StatsT, PredictT>::getLeafStats() {
  if (step != Step::LeavesProcessed)
    throw std::runtime_error("Cannot provide bin stats without first "
                             "training the discretizer");
  return leafStats;
}
template <typename StatsT, typename PredictT>
std::vector<size_t> &
UnivariateDiscretizer<StatsT, PredictT>::getLeafNumSamples() {
  if (step != Step::LeavesProcessed)
    throw std::runtime_error(
        "Cannot provide bin number of samples without first "
        "training the discretizer");
  return leafNumSamples;
}
template <typename StatsT, typename PredictT>
std::vector<double> &
UnivariateDiscretizer<StatsT, PredictT>::getLeafNodeWeights() {
  if (step != Step::LeavesProcessed)
    throw std::runtime_error(
        "Cannot provide bin node weights without first training the discretizer");
  return leafNodeWeights;
}
