/**
 * @file CategoricalDiscretizer.tpp
 * @brief Template implementation for ``CategoricalDiscretizer`` training and routing.
 */

#include "../algorithms/TreeBuilder.h"
#include "Splitters/CategoricalSplitter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

template <typename StatsT, typename PredictT>
size_t CategoricalDiscretizer<StatsT, PredictT>::dominantActiveCategory(
    const arma::fmat &X, const std::vector<size_t> &samples,
    const std::vector<size_t> &availableCategories) const {
  for (size_t feat : availableCategories) {
    bool allActive = true;
    for (size_t i : samples) {
      if (!isActive(X(feat, i))) {
        allActive = false;
        break;
      }
    }
    if (allActive)
      return feat;
  }
  return SIZE_MAX;
}

template <typename StatsT, typename PredictT>
size_t CategoricalDiscretizer<StatsT, PredictT>::appendLeaf(
    const std::vector<size_t> &samples, size_t categoryFeature) {
  LeafRecord leaf;
  leaf.samples = samples;
  leaf.categoryFeature = categoryFeature;
  leaves_.push_back(std::move(leaf));
  return leaves_.size() - 1;
}

template <typename StatsT, typename PredictT>
void CategoricalDiscretizer<StatsT, PredictT>::finalizeNodeAsLeaf(
    const arma::fmat &X, size_t nodeId, const std::vector<size_t> &samples,
    const std::vector<size_t> &availableCategories) {
  const size_t cat = dominantActiveCategory(X, samples, availableCategories);
  RoutingNode leafNode;
  leafNode.isLeaf = true;
  leafNode.leafBin = appendLeaf(samples, cat);
  routing_[nodeId] = leafNode;
}

template <typename StatsT, typename PredictT>
void CategoricalDiscretizer<StatsT, PredictT>::buildTree(
    const arma::fmat &X, CategoricalSplitter<StatsT, PredictT> &splitter,
    size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes) {
  this->resetTrainedOutputs();
  routing_.clear();
  leaves_.clear();
  routing_.push_back(RoutingNode{});
  step = Step::Untrained;

  CategoricalSplitCandidate root = splitter.makeRoot();
  root.nodeId = 0;
  root.score = splitter.score(root.samples);

  const double eps = std::numeric_limits<double>::epsilon();
  std::unordered_map<size_t, CategoricalSplitCandidate> openNodes;

  auto findBest = [&](CategoricalSplitCandidate &node, size_t mls) -> bool {
    if (node.isActiveLeafBranch)
      return false;
    return splitter.findBestSplit(node, mls);
  };

  auto makeChildren = [&](CategoricalSplitCandidate &parent) {
    return splitter.makeChildren(parent);
  };

  auto commitSplit = [&](CategoricalSplitCandidate &parent,
                         std::vector<CategoricalSplitCandidate> &children) {
    openNodes.erase(parent.nodeId);

    CategoricalSplitCandidate &inactive = children[0];
    CategoricalSplitCandidate &active = children[1];

    const size_t activeLeafBin =
        appendLeaf(active.samples, parent.splitFeature);

    routing_[parent.nodeId].isLeaf = false;
    routing_[parent.nodeId].splitFeature = parent.splitFeature;
    routing_[parent.nodeId].activeLeafBin = activeLeafBin;

    inactive.nodeId = routing_.size();
    routing_.push_back(RoutingNode{});
    routing_[parent.nodeId].inactiveChild = inactive.nodeId;

    CategoricalSplitCandidate probe = inactive;
    const bool canExpand =
        probe.score > eps &&
        (maxDepth == 0 || probe.height < maxDepth) &&
        splitter.findBestSplit(probe, minLeafSize);
    if (!canExpand) {
      finalizeNodeAsLeaf(X, inactive.nodeId, inactive.samples,
                         inactive.availableCategoryFeatures);
    } else {
      routing_[inactive.nodeId].isLeaf = false;
      openNodes[inactive.nodeId] = inactive;
    }
  };

  if (root.score <= eps || !splitter.findBestSplit(root, minLeafSize)) {
    finalizeNodeAsLeaf(X, root.nodeId, root.samples,
                       root.availableCategoryFeatures);
    step = Step::FitTree;
    this->numLeaves_ = leaves_.size();
    return;
  }

  TreeBuilder<CategoricalSplitCandidate> treeBuilder(
      minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
  treeBuilder.buildTree(root, findBest, makeChildren, commitSplit);

  for (const auto &[nodeId, node] : openNodes)
    finalizeNodeAsLeaf(X, nodeId, node.samples, node.availableCategoryFeatures);

  step = Step::FitTree;
  this->numLeaves_ = leaves_.size();
}

template <typename StatsT, typename PredictT>
void CategoricalDiscretizer<StatsT, PredictT>::appendNanRoutingBin() {
  this->inSampleDiscretizations_.push_back({});
  const size_t statsDim =
      this->leafStats_.empty() ? 0 : this->leafStats_.front().size();
  this->leafStats_.push_back(std::vector<StatsT>(statsDim, StatsT{0}));
  this->leafNumSamples_.push_back(0);
  this->leafNodeWeights_.push_back(0.0);
  binPredictions_.push_back(PredictT{});
}

template <typename StatsT, typename PredictT>
void CategoricalDiscretizer<StatsT, PredictT>::processLeaves(
    CategoricalSplitter<StatsT, PredictT> &splitter) {
  if (step != Step::FitTree)
    throw std::runtime_error("tree must be fit before leaves are processed");

  this->inSampleDiscretizations_.clear();
  binPredictions_.clear();
  this->leafStats_.clear();
  this->leafNumSamples_.clear();
  this->leafNodeWeights_.clear();

  for (const auto &leaf : leaves_) {
    this->inSampleDiscretizations_.push_back(leaf.samples);
    binPredictions_.push_back(splitter.predict(leaf.samples));
    this->leafStats_.push_back(splitter.statsForSamples(leaf.samples));
    this->leafNumSamples_.push_back(leaf.samples.size());
    this->leafNodeWeights_.push_back(splitter.totalWeight(leaf.samples));
  }

  step = Step::LeavesProcessed;
  appendNanRoutingBin();
  this->markTrained();
}

template <typename StatsT, typename PredictT>
size_t CategoricalDiscretizer<StatsT, PredictT>::routeOne(const arma::fmat &X,
                                                          arma::uword col) const {
  this->ensureTrained();
  size_t nodeId = 0;
  while (!routing_[nodeId].isLeaf) {
    const RoutingNode &n = routing_[nodeId];
    if (isActive(X(n.splitFeature, col)))
      return n.activeLeafBin;
    nodeId = n.inactiveChild;
  }
  return routing_[nodeId].leafBin;
}

template <typename StatsT, typename PredictT>
void CategoricalDiscretizer<StatsT, PredictT>::transform(
    const arma::fmat &X, arma::Row<size_t> &binLoc) const {
  this->ensureTrained();
  binLoc.set_size(X.n_cols);
  const size_t nanBin = nanBinIndex();
  for (arma::uword c = 0; c < X.n_cols; ++c) {
    bool anyActive = false;
    for (size_t feat : featureIndices_) {
      if (isActive(X(feat, c))) {
        anyActive = true;
        break;
      }
    }
    if (!anyActive) {
      binLoc(c) = nanBin;
      continue;
    }
    binLoc(c) = routeOne(X, c);
  }
}

template <typename StatsT, typename PredictT>
size_t CategoricalDiscretizer<StatsT, PredictT>::routeToBin(
    const std::vector<float> &featureValues) const {
  if (featureValues.size() != featureIndices_.size())
    throw std::invalid_argument(
        "featureValues length must match trained featureIndices count");
  this->ensureTrained();

  bool anyActive = false;
  for (float v : featureValues) {
    if (isActive(v)) {
      anyActive = true;
      break;
    }
  }
  if (!anyActive)
    return nanBinIndex();

  size_t nodeId = 0;
  while (!routing_[nodeId].isLeaf) {
    const RoutingNode &n = routing_[nodeId];
    const auto it = std::find(featureIndices_.begin(), featureIndices_.end(),
                              n.splitFeature);
    if (it == featureIndices_.end())
      throw std::runtime_error("routing split feature not in featureIndices");
    const size_t pos = static_cast<size_t>(it - featureIndices_.begin());
    if (isActive(featureValues[pos]))
      return n.activeLeafBin;
    nodeId = n.inactiveChild;
  }
  return routing_[nodeId].leafBin;
}
