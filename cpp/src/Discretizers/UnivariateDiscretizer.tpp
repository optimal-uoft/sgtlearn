/**
 * @file UnivariateDiscretizer.tpp
 * @brief Template implementation for ``UnivariateDiscretizer`` training and ``transform``.
 */

#include "../algorithms/TreeBuilder.h"
#include "../algorithms/missing_values.h"

#include <algorithm>
#include <iterator>

template <typename StatsT, typename PredictT>
void UnivariateDiscretizer<StatsT, PredictT>::processLeaves(
    arma::uvec sortedOrder, Splitter<StatsT, PredictT> &splitter) {
  if (step != Step::FitTree)
    throw std::runtime_error("tree must be fit before leaves are processed");
  this->inSampleDiscretizations_.clear();
  binPredictions.clear();
  thresholds_.clear();
  this->leafStats_.clear();
  this->leafNumSamples_.clear();
  this->leafNodeWeights_.clear();
  for (auto &[_, leaf] : leaves) {
    this->inSampleDiscretizations_.push_back(
        arma::conv_to<std::vector<size_t>>::from(
            sortedOrder.subvec(leaf.start, leaf.end)));
    binPredictions.push_back(splitter.predict(leaf));
    this->leafStats_.push_back(splitter.getStats(leaf));
    this->leafNumSamples_.push_back(leaf.numSamples);
    this->leafNodeWeights_.push_back(leaf.nodeWeight);
    thresholds_.push_back(leaf.routingThreshold);
  }
  step = Step::LeavesProcessed;
  appendNanRoutingBin();
  this->markTrained();
}

template <typename StatsT, typename PredictT>
void UnivariateDiscretizer<StatsT, PredictT>::appendNanRoutingBin() {
  this->inSampleDiscretizations_.push_back(nanInSampleIndices_);
  this->leafStats_.push_back(nanStats_);
  this->leafNumSamples_.push_back(nanNumSamples_);
  this->leafNodeWeights_.push_back(nanNodeWeight_);
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
  this->numLeaves_ = leaves.size();
}

template <typename StatsT, typename PredictT>
size_t UnivariateDiscretizer<StatsT, PredictT>::routeToBin(
    const std::vector<float> &featureValues) const {
  this->ensureTrained();
  if (featureValues.empty())
    throw std::runtime_error("featureValues is empty");
  if (!missing_values::is_finite(featureValues[0]))
    return nanBinIndex();
  const auto it =
      std::lower_bound(thresholds_.begin(), thresholds_.end(),
                       static_cast<double>(featureValues[0]));
  return static_cast<size_t>(std::distance(thresholds_.begin(), it));
}

template <typename StatsT, typename PredictT>
void UnivariateDiscretizer<StatsT, PredictT>::transform(const arma::fmat &X,
                                                      arma::Row<size_t> &binLoc) const {
  this->ensureTrained();

  if (feature >= X.n_rows)
    throw std::invalid_argument("Trained feature must be an index < X.n_rows");
  binLoc = arma::Row<size_t>(X.n_cols);
  const arma::frowvec featureRow = X.row(feature);
  const size_t nanBin = nanBinIndex();
  for (size_t col = 0; col < X.n_cols; ++col) {
    const float value = featureRow(col);
    if (!missing_values::is_finite(value)) {
      binLoc(col) = nanBin;
      continue;
    }
    const auto it =
        std::lower_bound(thresholds_.begin(), thresholds_.end(), value);
    binLoc(col) = static_cast<size_t>(std::distance(thresholds_.begin(), it));
  }
}

template <typename StatsT, typename PredictT>
std::vector<PredictT> &UnivariateDiscretizer<StatsT, PredictT>::getBinPredictions() {
  this->ensureTrained();
  return binPredictions;
}
