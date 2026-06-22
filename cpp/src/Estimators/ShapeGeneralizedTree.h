#pragma once

/**
 * @file Estimators/ShapeGeneralizedTree.h
 * @brief Common fitted-tree storage for shape-generalized estimators.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"

#include <cstddef>
#include <vector>

/**
 * Fitted outer routing tree shared by classification and regression shape-
 * generalized estimators.
 *
 * Subclasses own task-specific fit/predict logic; this base holds the node
 * array and child index map used for routing at inference time.
 */
class ShapeGeneralizedTree {
public:
  virtual ~ShapeGeneralizedTree() = default;

  /** Read-only access to the fitted node array for introspection / export. */
  const std::vector<ShapeFunctionNode> &nodes() const { return nodes_; }

  /** Per-node child indices; empty inner vector at leaves. */
  const std::vector<std::vector<size_t>> &childIndices() const {
    return childIndices_;
  }

  /** Index of the root node (currently always 0 after fit). */
  size_t rootIndex() const { return rootIndex_; }

  /** Number of leaf nodes in the fitted outer tree. */
  size_t numLeaves() const;

  /** Total number of nodes (internal + leaf) in the fitted outer tree. */
  size_t numNodes() const;

  /** True if `fit` has completed successfully. */
  bool isFitted() const { return fitted_; }

protected:
  bool fitted_ = false;

  std::vector<ShapeFunctionNode> nodes_;
  /**
   * For node i, childIndices_[i][p] is the node index of child partition p.
   * Empty for leaves.
   */
  std::vector<std::vector<size_t>> childIndices_;
  size_t rootIndex_ = 0;
};
