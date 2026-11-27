#pragma once

/**
 * @file Discretizers/pair/PairDiscretizerBase.h
 * @brief Shared routing-tree, axis validation, and tree-growth machinery for
 *        ``PairClassificationDiscretizer`` and ``PairRegressionDiscretizer``.
 */

#include "Domain/FeatureInfo.h"
#include "Domain/LearningCriterion.h"
#include "algorithms/missing_values.h"

#include <algorithm>
#include <armadillo>
#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

struct PairRoutingTreeNode {
  bool isLeaf = true;
  size_t rawFeature = 0;
  size_t featurePosition = 0;
  FeatureType featureType = FeatureType::Continuous;
  double threshold = 0.0;
  size_t left = 0;
  size_t right = 0;
  size_t missing = 0;
  size_t bin = 0;
};

struct PairBuildNode {
  PairRoutingTreeNode routing;
  std::vector<size_t> samples;
  std::vector<std::vector<double>> stats;
  double weight = 0.0;
  double impurity = 0.0;
  size_t depth = 0;
};

struct PairSplit {
  bool found = false;
  size_t featurePosition = 0;
  size_t rawFeature = 0;
  FeatureType featureType = FeatureType::Continuous;
  double threshold = 0.0;
  double gain = 0.0;
  std::vector<size_t> left;
  std::vector<size_t> right;
  std::vector<size_t> missing;
};

inline std::vector<size_t>
pairRootBinAssignments(const std::vector<PairRoutingTreeNode> &tree,
                       size_t numLeaves, size_t missingBranch) {
  if (tree.empty() || tree[0].isLeaf)
    return {};
  std::vector<size_t> assignments(numLeaves, 0);
  const auto assign = [&](auto &&self, size_t index, size_t branch) -> void {
    const auto &node = tree[index];
    if (node.isLeaf) {
      assignments[node.bin] = branch;
      return;
    }
    self(self, node.left, branch);
    self(self, node.right, branch);
    if (node.missing != node.left && node.missing != node.right)
      self(self, node.missing, branch);
  };
  assign(assign, tree[0].left, 0);
  assign(assign, tree[0].right, 1);
  if (tree[0].missing != tree[0].left && tree[0].missing != tree[0].right)
    assign(assign, tree[0].missing, missingBranch);
  return assignments;
}

/** Shared non-virtual implementation mixin for pair CART discretizers. */
class PairDiscretizerMixin {
public:
  const std::vector<PairRoutingTreeNode> &routingTree() const { return tree_; }
  const std::array<FeatureInfo, 2> &axes() const { return axes_; }

protected:
  PairDiscretizerMixin(LearningCriterion criterion, FeatureInfo first,
                       FeatureInfo second)
      : criterion_(criterion), axes_{std::move(first), std::move(second)} {
    axisOffsets_[1] = axes_[0].indices.n_elem;
    axisOffsets_[2] = axisOffsets_[1] + axes_[1].indices.n_elem;
    if (axisOffsets_[1] == 0 || axisOffsets_[2] == axisOffsets_[1])
      throw std::invalid_argument("pair CART logical features cannot be empty");
    for (const FeatureInfo &axis : axes_)
      if (axis.type == FeatureType::Continuous && axis.indices.n_elem != 1)
        throw std::invalid_argument(
            "pair CART continuous logical features require one column");
    routingFeatures_ = arma::join_cols(axes_[0].indices, axes_[1].indices);
  }

  arma::Row<float>
  preparePairWeights(const arma::fmat &X, const arma::uvec &features,
                     const arma::Row<float> &sampleWeights) const {
    if (features.n_elem != routingFeatures_.n_elem)
      throw std::invalid_argument("invalid pair CART training shapes");
    arma::Row<float> weights = sampleWeights;
    if (weights.n_elem == 0)
      weights.ones(X.n_cols);
    if (weights.n_elem != X.n_cols)
      throw std::invalid_argument("pair CART sample weight length mismatch");
    for (size_t rawFeature : routingFeatures_)
      if (rawFeature >= X.n_rows)
        throw std::invalid_argument("pair CART feature out of range");
    return weights;
  }

  bool axisMissing(size_t axis, const arma::fmat &X, size_t sample) const {
    if (axes_[axis].type == FeatureType::Continuous)
      return !missing_values::is_finite(X(axes_[axis].indices(0), sample));
    for (size_t raw : axes_[axis].indices)
      if (X(raw, sample) >= 0.5F)
        return false;
    return true;
  }

  bool axisMissing(size_t axis, const std::vector<float> &values) const {
    if (axes_[axis].type == FeatureType::Continuous)
      return !missing_values::is_finite(values[axisOffsets_[axis]]);
    for (size_t pos = axisOffsets_[axis]; pos < axisOffsets_[axis + 1]; ++pos)
      if (values[pos] >= 0.5F)
        return false;
    return true;
  }

  size_t routingPosition(size_t rawFeature) const {
    const auto it =
        std::find(routingFeatures_.begin(), routingFeatures_.end(), rawFeature);
    if (it == routingFeatures_.end())
      throw std::runtime_error("pair CART routing feature not found");
    return static_cast<size_t>(it - routingFeatures_.begin());
  }

  size_t routeValues(const std::vector<float> &values) const {
    if (values.size() != routingFeatures_.n_elem)
      throw std::invalid_argument(
          "pair router values do not match routing features");
    size_t index = 0;
    while (!tree_[index].isLeaf) {
      const PairRoutingTreeNode &node = tree_[index];
      if (axisMissing(node.featurePosition, values)) {
        index = node.missing;
        continue;
      }
      const float value = values[routingPosition(node.rawFeature)];
      index = node.featureType == FeatureType::Categorical
                  ? (value >= 0.5F ? node.right : node.left)
                  : (value <= node.threshold ? node.left : node.right);
    }
    return tree_[index].bin;
  }

  void transformRows(const arma::fmat &X, arma::Row<size_t> &binLoc) const {
    binLoc.set_size(X.n_cols);
    std::vector<float> values(routingFeatures_.n_elem);
    for (arma::uword i = 0; i < X.n_cols; ++i) {
      for (size_t j = 0; j < routingFeatures_.n_elem; ++j)
        values[j] = X(routingFeatures_(j), i);
      binLoc(i) = routeValues(values);
    }
  }

  template <typename MakeNodeFn, typename FindBestSplitFn>
  void buildPairTree(
      arma::uword nSamples, double minGainSplit, size_t maxDepth,
      size_t maxLeafNodes, MakeNodeFn &&makeNode,
      FindBestSplitFn &&findBestSplit,
      std::vector<std::vector<size_t>> &inSampleDiscretizations,
      std::vector<std::vector<std::vector<double>>> &leafStats,
      std::vector<size_t> &leafNumSamples, std::vector<double> &leafNodeWeights,
      size_t &numLeaves) {
    tree_.clear();
    std::vector<size_t> all(nSamples);
    std::iota(all.begin(), all.end(), 0);
    std::vector<PairBuildNode> nodes;
    nodes.push_back(makeNode(std::move(all), 0));
    const double totalWeight = nodes.front().weight;
    size_t finiteLeafCount = 1;

    const auto splitNode = [&](size_t index, PairSplit split) {
      const size_t depth = nodes[index].depth;
      const bool missingGoesLeft = split.left.size() >= split.right.size();
      nodes[index].routing.isLeaf = false;
      nodes[index].routing.featurePosition = split.featurePosition;
      nodes[index].routing.rawFeature = split.rawFeature;
      nodes[index].routing.featureType = split.featureType;
      nodes[index].routing.threshold = split.threshold;
      nodes[index].routing.left = nodes.size();
      nodes.push_back(makeNode(std::move(split.left), depth + 1));
      nodes[index].routing.right = nodes.size();
      nodes.push_back(makeNode(std::move(split.right), depth + 1));
      if (split.missing.empty()) {
        nodes[index].routing.missing = missingGoesLeft
                                           ? nodes[index].routing.left
                                           : nodes[index].routing.right;
      } else {
        nodes[index].routing.missing = nodes.size();
        nodes.push_back(makeNode(std::move(split.missing), depth + 1));
      }
      ++finiteLeafCount;
    };

    if (maxLeafNodes == 0) {
      const auto grow = [&](auto &&self, size_t index) -> void {
        if (maxDepth != 0 && nodes[index].depth >= maxDepth)
          return;
        PairSplit split = findBestSplit(nodes[index], totalWeight);
        if (!split.found ||
            split.gain + std::numeric_limits<double>::epsilon() < minGainSplit)
          return;
        splitNode(index, std::move(split));
        self(self, nodes[index].routing.left);
        self(self, nodes[index].routing.right);
        if (nodes[index].routing.missing != nodes[index].routing.left &&
            nodes[index].routing.missing != nodes[index].routing.right)
          self(self, nodes[index].routing.missing);
      };
      grow(grow, 0);
    } else {
      while (finiteLeafCount < maxLeafNodes) {
        size_t bestIndex = nodes.size();
        PairSplit best;
        for (size_t i = 0; i < nodes.size(); ++i) {
          if (!nodes[i].routing.isLeaf ||
              (maxDepth != 0 && nodes[i].depth >= maxDepth))
            continue;
          PairSplit candidate = findBestSplit(nodes[i], totalWeight);
          if (!candidate.found ||
              candidate.gain + std::numeric_limits<double>::epsilon() <
                  minGainSplit)
            continue;
          if (bestIndex == nodes.size() || candidate.gain > best.gain) {
            bestIndex = i;
            best = std::move(candidate);
          }
        }
        if (bestIndex == nodes.size())
          break;
        splitNode(bestIndex, std::move(best));
      }
    }

    for (PairBuildNode &node : nodes) {
      if (!node.routing.isLeaf)
        continue;
      node.routing.bin = inSampleDiscretizations.size();
      inSampleDiscretizations.push_back(std::move(node.samples));
      leafStats.push_back(std::move(node.stats));
      leafNumSamples.push_back(inSampleDiscretizations.back().size());
      leafNodeWeights.push_back(node.weight);
    }
    tree_.reserve(nodes.size());
    for (const PairBuildNode &node : nodes)
      tree_.push_back(node.routing);
    numLeaves = leafStats.size();
  }

  LearningCriterion criterion_;
  std::array<FeatureInfo, 2> axes_;
  arma::uvec routingFeatures_;
  std::array<size_t, 3> axisOffsets_{};
  std::vector<PairRoutingTreeNode> tree_;
};
