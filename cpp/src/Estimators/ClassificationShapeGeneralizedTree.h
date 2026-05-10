#pragma once

/**
 * @file Estimators/ClassificationShapeGeneralizedTree.h
 * @brief Multivariate shape-generalized classification tree (outer ``TreeBuilder`` + inner branching fits).
 */

#include "Domain/LearningCriterion.h"
#include "algorithms/ShapeGeneralizedTreeNodeBase.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"
#include "algorithms/TreeBuilder.h"

#include <armadillo>
#include <compare>
#include <cstddef>
#include <vector>

/**
 * Classification outer-tree node: routing state from ShapeGeneralizedTreeNodeBase
 * plus a per-class histogram at leaves.
 *
 * Routing and histogram sizes come from the vectors (innerThresholds,
 * binToPartition, childSampleBounds, leafClassCounts). Class cardinality is a
 * model-level property, not duplicated on each node.
 *
 * @see TreeBuilder — requires score, informationGain, height, and a weak
 *      ordering consistent with SplitCandidate (higher informationGain first
 *      in the best-first heap).
 */
struct ClassificationShapeGeneralizedNode : ShapeGeneralizedTreeNodeBase {
  /** Leaf-only: counts per class label (encoding matches training y). */
  std::vector<size_t> leafClassCounts;

  std::weak_ordering
  operator<=>(const ClassificationShapeGeneralizedNode &o) const {
    return std::compare_weak_order_fallback(informationGain,
                                            o.informationGain);
  }

  /** Equality matches the heap ordering key (informationGain), not full struct equality. */
  bool operator==(const ClassificationShapeGeneralizedNode &o) const {
    return informationGain == o.informationGain;
  }
};

/**
 * Shape-Generalized Tree, classification variant.
 *
 * Responsibilities (by phase):
 * - **Outer growth** (`TreeBuilder`): best-first or depth-first expansion using
 *   `findBestSplitNode` / `makeChildrenNode` / `commitSplitNode`.
 * - **Per-node split search**: delegate to `fitShapeBranch` (inner discretizer
 *   + coordinate descent over bins); see algorithms/ShapeBranchingFit.cpp.
 * - **Leaf state**: `fillLeafHistogram`, `impurityAtRange` for Gini/entropy.
 * - **Inference**: `predict` / `predictProba` walk childIndices_ using
 *   `routeFeatureValueToPartition`.
 *
 * At every outer-tree node, for each candidate feature:
 *   1. Train a UnivariateClassificationDiscretizer (inner tree) on the
 *      node's samples to obtain a bin index per sample and per-bin class
 *      counts.
 *   2. Build a classification BranchAssignment from those bin stats and run
 *      coordinateDescent to find a bin -> partition mapping that minimises
 *      the criterion impurity over `numPartitions` partitions.
 *   3. Score the resulting partition impurity.
 *
 * The best-scoring feature wins; its inner discretizer + bin->partition
 * mapping become the routing rule for that node, producing `numPartitions`
 * children. Inner-node fitting matches the Python `BranchingTree` pattern
 * (discretizer + coordinate descent over bins); the outer loop uses
 * `TreeBuilder` like Python's heap over `best_impurity_decrease`.
 *
 * Inputs use Armadillo's column-major convention: X has shape
 * (numFeatures, numSamples); each column is one sample.
 *
 * @note Only `LearningCriterion::Entropy` and `LearningCriterion::Gini` are
 *       accepted; other criteria throw from the constructor.
 *
 * **Regression reuse:** outer topology and `ShapeGeneralizedTreeNodeBase` are
 * task-agnostic; a regression trainer would swap leaf payloads, impurity,
 * `fitShapeBranch` for regression discretizers + `makeRegressionBranchAssignment`,
 * and prediction (e.g. leaf mean).
 */
class ClassificationShapeGeneralizedTree {
public:
  /**
   * @param criterion       impurity for inner splits, partition scoring, and
   *                        coordinate descent (Entropy or Gini).
   * @param numClasses      number of distinct class labels in [0, numClasses).
   * @param numPartitions   fan-out of every internal routing node (>= 2).
   * @param outerParams     tree-building params for the outer routing tree.
   * @param innerParams     tree-building params for the per-feature inner
   *                        univariate discretizer.
   * @param cdParams        coordinate-descent loop parameters.
   */
  ClassificationShapeGeneralizedTree(
      LearningCriterion criterion, size_t numClasses, size_t numPartitions,
      TreeBuildingParams outerParams = {},
      TreeBuildingParams innerParams = {},
      CoordinateDescentParams cdParams = {});

  ~ClassificationShapeGeneralizedTree() = default;

  ClassificationShapeGeneralizedTree(
      const ClassificationShapeGeneralizedTree &) = delete;
  ClassificationShapeGeneralizedTree &
  operator=(const ClassificationShapeGeneralizedTree &) = delete;
  ClassificationShapeGeneralizedTree(ClassificationShapeGeneralizedTree &&) noexcept =
      default;
  ClassificationShapeGeneralizedTree &
  operator=(ClassificationShapeGeneralizedTree &&) noexcept = default;

  /**
   * Fit the routing tree.
   *
   * @param X         (numFeatures, numSamples) column-major; one sample per
   *                  column.
   * @param features  feature indices considered as routing candidates at
   *                  every node. Non-const because internal sort scratch
   *                  may reuse it.
   * @param y         (numSamples,) integer class labels in [0, numClasses).
   *
   * @throws std::invalid_argument on shape / label-range mismatch.
   */
  void fit(const arma::fmat &X, arma::uvec &features,
           const arma::Row<size_t> &y);

  /** Hard class predictions, shape (numSamples,). */
  arma::Row<size_t> predict(const arma::fmat &X) const;

  /** Class probabilities, shape (numClasses, numSamples). */
  arma::fmat predictProba(const arma::fmat &X) const;

  /** Number of leaf nodes in the fitted outer tree. */
  size_t numLeaves() const;

  /** Total number of nodes (internal + leaf) in the fitted outer tree. */
  size_t numNodes() const;

  /** True if `fit` has completed successfully. */
  bool isFitted() const;

  // TODO: flat introspection accessors for graphviz / text export, e.g.
  //   featureOf(node), thresholdsOf(node), binToPartitionOf(node),
  //   childrenOf(node), classCountsOf(leaf), depthOf(node).

private:
  LearningCriterion criterion_;
  size_t numClasses_;
  size_t numPartitions_;
  TreeBuildingParams outerParams_;
  TreeBuildingParams innerParams_;
  CoordinateDescentParams cdParams_;
  bool fitted_ = false;

  std::vector<ClassificationShapeGeneralizedNode> nodes_;
  /**
   * For node i, childIndices_[i][p] is the node index of child partition p.
   * Empty for leaves.
   */
  std::vector<std::vector<size_t>> childIndices_;
  size_t rootIndex_ = 0;

  /** Outer routing expansion; `fit` passes split logic via buildTree callbacks. */
  TreeBuilder<ClassificationShapeGeneralizedNode> outerTreeBuilder_;

  /** Column permutation of training data; used only inside `fit`. */
  arma::uvec sampleOrder_;

  double impurityAtRange(size_t begin, size_t end,
                         const arma::Row<size_t> &y) const;

  void fillLeafHistogram(ClassificationShapeGeneralizedNode &node,
                         const arma::Row<size_t> &y) const;

  bool findBestSplitNode(ClassificationShapeGeneralizedNode &node,
                         size_t minLeafSize, const arma::fmat &X,
                         const arma::Row<size_t> &y, const arma::uvec &features);

  std::vector<ClassificationShapeGeneralizedNode>
  makeChildrenNode(ClassificationShapeGeneralizedNode &parent,
                   const arma::fmat &X, const arma::Row<size_t> &y);

  void commitSplitNode(ClassificationShapeGeneralizedNode &parent,
                       std::vector<ClassificationShapeGeneralizedNode> &children);
};
