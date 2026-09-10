#pragma once

#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"

#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>

/** Outer induction has constant regularization costs; inner CART keeps TreeBuilder. */
class OuterTreeBuilder {
public:
  static constexpr double eps = std::numeric_limits<double>::epsilon();

  explicit OuterTreeBuilder(const TreeBuildingParams &params) : params_(params) {}

  void buildTree(
      ShapeFunctionNode &root,
      const std::function<std::vector<ShapeFunctionNode>(ShapeFunctionNode &, size_t, size_t)> &findSplits,
      const std::function<std::vector<ShapeFunctionNode>(ShapeFunctionNode &)> &makeChildren,
      const std::function<void(ShapeFunctionNode &, std::vector<ShapeFunctionNode> &)> &commitSplit) {
    // Each node keeps its discovered alternatives, sorted best first.
    std::vector<std::vector<ShapeFunctionNode>> frontier;
    const auto lowerPriority = [](const auto &a, const auto &b) {
      return a.front() < b.front();
    };
    const size_t configuredArity = root.numPartitions;
    size_t numLeaves = 1;
    const auto discover = [&](ShapeFunctionNode &node) {
      const size_t maxArity = params_.maxLeafNodes == 0
          ? node.numPartitions : std::min(node.numPartitions, params_.maxLeafNodes - numLeaves + 1);
      if ((params_.maxDepth != 0 && node.height >= params_.maxDepth) ||
          node.sampleIndices.n_elem < 2 * params_.minLeafSize)
        return;
      auto candidates = findSplits(node, params_.minLeafSize, maxArity);
      std::erase_if(candidates, [maxArity](const auto &candidate) {
        return candidate.numPartitions > maxArity ||
            !std::isfinite(candidate.regularizedGain) || candidate.regularizedGain <= eps;
      });
      if (candidates.empty())
        return;
      std::sort(candidates.begin(), candidates.end(), std::greater<>());
      frontier.push_back(std::move(candidates));
      std::push_heap(frontier.begin(), frontier.end(), lowerPriority);
    };
    if (params_.maxLeafNodes == 0 || numLeaves < params_.maxLeafNodes)
      discover(root);
    while (!frontier.empty() &&
           (params_.maxLeafNodes == 0 || numLeaves < params_.maxLeafNodes)) {
      std::pop_heap(frontier.begin(), frontier.end(), lowerPriority);
      auto split = std::move(frontier.back().front());
      frontier.pop_back();
      auto children = makeChildren(split);
      commitSplit(split, children);
      numLeaves += children.size() - 1;
      if (params_.maxLeafNodes != 0 &&
          params_.maxLeafNodes - numLeaves + 1 < configuredArity) {
        const size_t maxArity = params_.maxLeafNodes - numLeaves + 1;
        for (auto &candidates : frontier)
          std::erase_if(candidates, [maxArity](const auto &candidate) {
            return candidate.numPartitions > maxArity;
          });
        std::erase_if(frontier, [](const auto &candidates) { return candidates.empty(); });
        std::make_heap(frontier.begin(), frontier.end(), lowerPriority);
      }
      if (params_.maxLeafNodes == 0 || numLeaves < params_.maxLeafNodes)
        for (auto &child : children)
          discover(child);
    }
  }

private:
  TreeBuildingParams params_;
};
