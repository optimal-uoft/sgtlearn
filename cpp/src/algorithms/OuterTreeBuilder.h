#pragma once

#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"

#include <cmath>
#include <functional>
#include <limits>
#include <queue>

/** Outer induction has constant regularization costs; inner CART keeps TreeBuilder. */
class OuterTreeBuilder {
public:
  static constexpr double eps = std::numeric_limits<double>::epsilon();

  explicit OuterTreeBuilder(const TreeBuildingParams &params) : params_(params) {}

  void buildTree(
      ShapeFunctionNode &root,
      const std::function<bool(ShapeFunctionNode &, size_t)> &findBestSplit,
      const std::function<std::vector<ShapeFunctionNode>(ShapeFunctionNode &)> &makeChildren,
      const std::function<void(ShapeFunctionNode &, std::vector<ShapeFunctionNode> &)> &commitSplit) {
    std::priority_queue<ShapeFunctionNode> frontier;
    const auto discover = [&](ShapeFunctionNode &node) {
      if ((params_.maxDepth == 0 || node.height < params_.maxDepth) &&
          node.sampleIndices.n_elem >= 2 * params_.minLeafSize &&
          findBestSplit(node, params_.minLeafSize) &&
          std::isfinite(node.regularizedGain) && node.regularizedGain > eps)
        frontier.push(node);
    };
    size_t numLeaves = 1;
    if (params_.maxLeafNodes == 0 || numLeaves < params_.maxLeafNodes)
      discover(root);
    while (!frontier.empty() &&
           (params_.maxLeafNodes == 0 || numLeaves < params_.maxLeafNodes)) {
      auto split = frontier.top();
      frontier.pop();
      auto children = makeChildren(split);
      commitSplit(split, children);
      numLeaves += children.size() - 1;
      if (params_.maxLeafNodes == 0 || numLeaves < params_.maxLeafNodes)
        for (auto &child : children)
          discover(child);
    }
  }

private:
  TreeBuildingParams params_;
};
