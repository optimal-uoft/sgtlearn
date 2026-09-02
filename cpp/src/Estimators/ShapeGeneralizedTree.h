#pragma once

/**
 * @file Estimators/ShapeGeneralizedTree.h
 * @brief Common fitted-tree storage for shape-generalized estimators.
 */

#include "Domain/LearningCriterion.h"
#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"

#include <armadillo>
#include <cstddef>
#include <vector>

/**
 * Fitted outer routing tree shared by classification and regression shape-
 * generalized estimators.
 *
 * Subclasses own task-specific fit/predict logic; this base holds the node
 * array, child index map, and shared hyperparameters used for routing at
 * inference time and post-fit refinement (TAO).
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

  /** Mutable node array for in-place routing refinement (e.g. TAO). */
  std::vector<ShapeFunctionNode> &mutableNodes() { return nodes_; }

  /** Mutable child index map for in-place routing refinement (e.g. TAO). */
  std::vector<std::vector<size_t>> &mutableChildIndices() {
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

  /**
   * Normalized per-feature importances (length = number of logical features
   * passed to ``fit``). One-to-one with that feature sequence. Valid only
   * after training completes; throws if the model is not fitted.
   */
  const arma::vec &featureImportance() const;

  /** Rebuild feature importances after an in-place routing refinement. */
  void refreshFeatureImportances();

  /** Impurity criterion this tree was constructed with. */
  virtual LearningCriterion criterion() const { return criterion_; }

  /** Fan-out used by every internal routing node. */
  size_t numPartitions() const { return numPartitions_; }

  /** Outer routing-tree hyperparameters (read-only). */
  const TreeBuildingParams &outerParams() const { return outerParams_; }

  /** Inner per-feature discretizer hyperparameters (read-only). */
  const TreeBuildingParams &innerParams() const { return innerParams_; }

protected:
  ShapeGeneralizedTree(LearningCriterion criterion, size_t numPartitions,
                       TreeBuildingParams outerParams,
                       TreeBuildingParams innerParams);

  bool fitted_ = false;

  LearningCriterion criterion_;
  size_t numPartitions_;
  TreeBuildingParams outerParams_;
  TreeBuildingParams innerParams_;

  std::vector<ShapeFunctionNode> nodes_;
  /**
   * For node i, childIndices_[i][p] is the node index of child partition p.
   * Empty for leaves.
   */
  std::vector<std::vector<size_t>> childIndices_;
  size_t rootIndex_ = 0;

  /** Normalized importances; written at end of ``fit``. */
  arma::vec featureImportance_;
  /** Running sum of node importances per logical feature during ``fit``. */
  arma::vec sumOfNodeImportancesByFeature_;
  /** Sum of all committed node importances during ``fit``. */
  double totalNodeImportanceSum_ = 0.0;
};
